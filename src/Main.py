# Main.py
import os
import sys
import argparse
import importlib.util
from datetime import datetime
import time
import json
import shutil
import signal

# Import the Trace generation and MCDC analysis components
import TraceGenerator as trace_generator
import MCDCAnalyzer as mcdc_analyzer

# Custom exception for the timeout
class TimeoutException(Exception):
    pass

# Handler function to raise the exception when the alarm signal is received
def timeout_handler(signum, frame):
    raise TimeoutException("File processing timed out.")

def main(input_folder: str, limit_mb: int | None, trace_only: bool, timeout_min: float | None, use_builtin_eval: bool): 
    pipeline_start_time = time.perf_counter()
    
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
    elif timeout_min is not None:
        print("Warning: Timeout functionality is not supported on this operating system. The --timeout-min flag will be ignored.", file=sys.stderr)
        timeout_min = None

    # Define a base directory for all outputs in the current working directory
    output_base_dir = "analysis_output"
    
    TRACES_DIR = os.path.join(output_base_dir, "traces")
    REPORTS_DIR = os.path.join(output_base_dir, "coverage_data")
    
    print(f"Saving all output to the '{output_base_dir}' directory in the current location.")
    
    os.makedirs(TRACES_DIR, exist_ok=True)
    if not trace_only:
        os.makedirs(REPORTS_DIR, exist_ok=True)
    
    abs_input_folder = os.path.abspath(input_folder)
    input_folder_name = os.path.basename(abs_input_folder)

    # Define and check for the trace completion marker
    top_level_trace_dir = os.path.join(TRACES_DIR, input_folder_name)
    trace_complete_marker = os.path.join(top_level_trace_dir, ".traces_complete")
    traces_are_pregenerated = os.path.exists(trace_complete_marker)

    print(f"\nStarting analysis pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not trace_only and traces_are_pregenerated:
        print(f"Marker file found. Assuming traces are pre-generated and skipping generation step.")
    if trace_only:
        print("Running in TRACE-ONLY mode. Analysis will be skipped.")
    if timeout_min:
        print(f"Per-file processing timeout set to {timeout_min} minutes.")
    print("-" * 60)

    # Gather all target Python files, excluding specific ones
    target_files = []
    excluded_files = ["Main.py", "MCDCAnalyzer.py", "TraceGenerator.py", "Tracing.py"]
    for root, _, files in os.walk(abs_input_folder):
        for file in files:
            if file.endswith(".py") and file not in excluded_files:
                target_files.append(os.path.join(root, file))

    if not target_files:
        print("No Python files found in the specified folder. Exiting.")
        return

    max_bytes = limit_mb * 1024 * 1024 if limit_mb is not None and limit_mb > 0 else None
    if max_bytes:
        print(f"Trace file size limit set to {limit_mb} MB ({max_bytes} bytes).")

    all_results = []
    total_files_to_analyze = len(target_files)
    timeout_seconds = int(timeout_min * 60) if timeout_min is not None and timeout_min > 0 else 0
    
    # Initialize counters for the new summary stats
    timeout_count = 0
    oversized_trace_count = 0
    
    for filepath in target_files:
        try:
            if timeout_seconds > 0:
                signal.alarm(timeout_seconds)

            relative_dir = os.path.relpath(os.path.dirname(filepath), abs_input_folder)
            module_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # Trace paths are always needed
            trace_subdir = os.path.join(TRACES_DIR, input_folder_name, relative_dir)
            os.makedirs(trace_subdir, exist_ok=True)
            trace_file_path = os.path.join(trace_subdir, f"{module_name}_trace.txt")

            # Report paths are only defined and created when not in trace-only mode
            report_file_path, json_report_path = None, None
            if not trace_only:
                report_subdir = os.path.join(REPORTS_DIR, input_folder_name, relative_dir)
                os.makedirs(report_subdir, exist_ok=True)
                report_file_path = os.path.join(report_subdir, f"{module_name}_analysis_report.txt")
                json_report_path = os.path.join(report_subdir, f"{module_name}_metrics.json")
            
            print(f"Processing: {filepath}")

            # Skipping the analysis if results already exist and not in trace-only mode
            if not trace_only and json_report_path and os.path.exists(json_report_path):
                print(f"  - Results already exist. Skipping analysis and loading metrics from cache.")
                with open(json_report_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                
                if os.path.exists(trace_file_path):
                     os.remove(trace_file_path)
            else:
                try:
                    # Conditionally skip trace generation if not in trace-only mode and a marker file exists.
                    should_generate_trace = True
                    if not trace_only and traces_are_pregenerated:
                        if os.path.exists(trace_file_path):
                            print(f"  - Found pre-generated trace. Skipping generation.")
                            should_generate_trace = False
                        else:
                            print(f"  - WARNING: Expected pre-generated trace file not found at '{trace_file_path}'. Skipping analysis for this file.", file=sys.stderr)
                            continue # Move to the next file

                    if should_generate_trace:
                        spec = importlib.util.spec_from_file_location(module_name, filepath)
                        target_module = importlib.util.module_from_spec(spec)
                        sys.path.insert(0, os.path.dirname(filepath))
                        spec.loader.exec_module(target_module)
                        sys.path.pop(0)

                        trace_generator.generate_trace(target_module, trace_file_path, max_bytes=max_bytes)
                    
                    if max_bytes is not None and os.path.exists(trace_file_path):
                        if os.path.getsize(trace_file_path) >= max_bytes:
                            oversized_trace_count += 1

                    # Handle empty generated trace files
                    if os.path.exists(trace_file_path) and os.path.getsize(trace_file_path) == 0:
                        print(f"  - WARNING: Empty trace file generated for {module_name}. Moving to 'empty traces'.")
                        empty_traces_dir = os.path.join(trace_subdir, "empty traces")
                        os.makedirs(empty_traces_dir, exist_ok=True)
                        new_trace_path = os.path.join(empty_traces_dir, os.path.basename(trace_file_path))
                        if os.path.exists(new_trace_path):
                            os.remove(new_trace_path)
                        os.rename(trace_file_path, new_trace_path)
                        print(f"  - Copying source file '{os.path.basename(filepath)}' to 'empty traces' for inspection.")
                        shutil.copy(filepath, empty_traces_dir)
                        print(f"-> Skipped analysis for {module_name} due to empty trace.")
                        print("-" * 60)
                        continue

                    if trace_only:
                        print(f"  - Trace generation complete. Skipping analysis.")
                        print("-" * 60)
                        continue

                    # Perform the MCDC analysis
                    metrics = mcdc_analyzer.analyze(
                        source_filepath=filepath,
                        trace_filepath=trace_file_path,
                        report_filepath=report_file_path,
                        json_report_filepath=json_report_path,
                        log_root_path=output_base_dir,
                        log_subfolder_name=input_folder_name,
                        use_builtin_eval=use_builtin_eval
                    )
                    
                    print(f"  - Analysis complete. Deleting temporary trace file...")
                    if os.path.exists(trace_file_path):
                        os.remove(trace_file_path)
                    
                except Exception as e:
                    print(f"-> ERROR processing {filepath}: {e}", file=sys.stderr)
                    print("-" * 60)
                    continue
                finally:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
            
            display_path = os.path.join(relative_dir, module_name) if relative_dir != '.' else module_name
            all_results.append({'file': display_path, **metrics})
            print(f"-> Successfully processed. Report at: {report_file_path}")
            print("-" * 60)
        
        except TimeoutException as e:
            timeout_count += 1
            print(f"-> TIMEOUT processing {filepath}: {e}", file=sys.stderr)
            print("-" * 60)
            continue
        finally:
            if timeout_seconds > 0:
                signal.alarm(0)

    # Print the final summary
    if not trace_only and all_results:
        print_summary(all_results, total_files_to_analyze, timeout_count, oversized_trace_count)
    # Output summary for trace-only mode
    elif trace_only:
        print("\nTrace generation finished for all files.")
        print(f"  - Programs Timed Out:        {timeout_count}")
        print(f"  - Trace Files > Size Limit:  {oversized_trace_count}")

        # Create a marker file to indicate that trace generation is complete.
        if not os.path.exists(top_level_trace_dir):
            os.makedirs(top_level_trace_dir)
        with open(trace_complete_marker, 'w', encoding='utf-8') as f:
            f.write(f"Traces generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nSuccessfully created trace completion marker at: '{trace_complete_marker}'")

    # Final timing
    pipeline_end_time = time.perf_counter()
    duration = pipeline_end_time - pipeline_start_time
    print(f"\nPipeline finished in {duration:.2f} seconds.")

def print_summary(results: list, total_files_analyzed: int, timeout_count: int, oversized_trace_count: int):
    """Prints a formatted summary of all analysis results."""
    
    num_processed = len(results)
    processed_percent = (num_processed / total_files_analyzed * 100) if total_files_analyzed > 0 else 0
    
    # Calculate number and percentage of fully MC/DC covered programs
    fully_covered_count = 0
    for res in results:
        is_fully_covered = True
        for metric in ['mcdc']:
            covered, total = res[metric]
            if total == 0:
                is_fully_covered = False
                break
            if total > 0 and covered < total:
                is_fully_covered = False
                break
        if is_fully_covered:
            fully_covered_count += 1
            
    fully_covered_percent = (fully_covered_count / total_files_analyzed * 100) if total_files_analyzed > 0 else 0
    
    print("\n" + "="*90)
    print(f"{'Overall Processing Summary':^90}")
    print("="*90)
    print(f"  - Total Programs Found:      {total_files_analyzed}")
    print(f"  - Programs Processed:        {num_processed} ({processed_percent:.2f}%)")
    print(f"  - Programs w/ 100% MC/DC Coverage: {fully_covered_count} ({fully_covered_percent:.2f}%)")
    print(f"  - Programs Timed Out:        {timeout_count}")
    print(f"  - Trace Files > Size Limit:  {oversized_trace_count}")


    print("\n" + "="*90)
    print(f"{'Detailed Coverage Summary':^90}")
    print("="*90)
    
    headers = ["File", "Branch", "Decision", "Condition", "MC/DC"]
    print(f"{headers[0]:<40} | {headers[1]:^10} | {headers[2]:^10} | {headers[3]:^12} | {headers[4]:^10}")
    print("-" * 90)

    totals = {
        'branch': {'covered': 0, 'total': 0}, 'decision': {'covered': 0, 'total': 0},
        'condition': {'covered': 0, 'total': 0}, 'mcdc': {'covered': 0, 'total': 0},
    }
    percentage_sums = {'branch': 0.0, 'decision': 0.0, 'condition': 0.0, 'mcdc': 0.0}

    for res in results:
        # Unpack coverage data
        b_cov, b_tot = res['branch']
        d_cov, d_tot = res['decision']
        c_cov, c_tot = res['condition']
        m_cov, m_tot = res['mcdc']
        
        # Update totals
        totals['branch']['covered'] += b_cov; totals['branch']['total'] += b_tot
        totals['decision']['covered'] += d_cov; totals['decision']['total'] += d_tot
        totals['condition']['covered'] += c_cov; totals['condition']['total'] += c_tot
        totals['mcdc']['covered'] += m_cov; totals['mcdc']['total'] += m_tot

        # Calculate percentages for this file
        b_pct = (b_cov / b_tot * 100) if b_tot > 0 else 0.0
        d_pct = (d_cov / d_tot * 100) if d_tot > 0 else 0.0
        c_pct = (c_cov / c_tot * 100) if c_tot > 0 else 0.0
        m_pct = (m_cov / m_tot * 100) if m_tot > 0 else 0.0

        # Update percentage sums for averages
        percentage_sums['branch'] += b_pct
        percentage_sums['decision'] += d_pct
        percentage_sums['condition'] += c_pct
        percentage_sums['mcdc'] += m_pct

        print(f"{res['file']:<40} | {b_pct:9.2f}% | {d_pct:9.2f}% | {c_pct:11.2f}% | {m_pct:9.2f}%")

    print("-" * 90)
    
    # Calculate overall percentages from totals
    b_total_pct = (totals['branch']['covered'] / totals['branch']['total'] * 100) if totals['branch']['total'] > 0 else 0.0
    d_total_pct = (totals['decision']['covered'] / totals['decision']['total'] * 100) if totals['decision']['total'] > 0 else 0.0
    c_total_pct = (totals['condition']['covered'] / totals['condition']['total'] * 100) if totals['condition']['total'] > 0 else 0.0
    m_total_pct = (totals['mcdc']['covered'] / totals['mcdc']['total'] * 100) if totals['mcdc']['total'] > 0 else 0.0

    print(f"{'OVERALL (from totals)':<40} | {b_total_pct:9.2f}% | {d_total_pct:9.2f}% | {c_total_pct:11.2f}% | {m_total_pct:9.2f}%")
    print(f"{'':<40} | ({totals['branch']['covered']}/{totals['branch']['total']})   | ({totals['decision']['covered']}/{totals['decision']['total']})   | ({totals['condition']['covered']}/{totals['condition']['total']})      | ({totals['mcdc']['covered']}/{totals['mcdc']['total']})")

    num_files = len(results)

    # Calculate average percentages
    b_avg_pct = percentage_sums['branch'] / num_files if num_files > 0 else 0
    d_avg_pct = percentage_sums['decision'] / num_files if num_files > 0 else 0
    c_avg_pct = percentage_sums['condition'] / num_files if num_files > 0 else 0
    m_avg_pct = percentage_sums['mcdc'] / num_files if num_files > 0 else 0
    
    print("-" * 90)
    print(f"{'AVERAGE (by file)':<40} | {b_avg_pct:9.2f}% | {d_avg_pct:9.2f}% | {c_avg_pct:11.2f}% | {m_avg_pct:9.2f}%")
    
    print("="*90)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run a full MCDC analysis pipeline on a folder of Python files.")
    parser.add_argument("folder", help="The path to the folder containing Python files to analyze.")
    parser.add_argument(
        "--limit-mb", 
        type=int, 
        help="The maximum size in megabytes for each trace file.", 
        default=None
    )
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="If set, only generate trace files and skip the analysis phase."
    )
    parser.add_argument(
        "--timeout-min",
        type=float,
        help="Timeout in minutes for processing each individual file.",
        default=None
    )
    parser.add_argument(
        "--use-builtin-eval",
        action="store_true",
        help="Use the built-in Python eval() function instead of the custom SafeEvaluator. WARNING: This can be insecure."
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found at '{args.folder}'", file=sys.stderr)
        sys.exit(1)
        
    main(args.folder, args.limit_mb, args.trace_only, args.timeout_min, args.use_builtin_eval) 