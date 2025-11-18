# test_mcdc_analysis.py
# This script should be located in the 'test/' directory.

import pytest
import json
import io
import logging
import os
import sys
import importlib.util

# --- Path Setup ---
# This script assumes it is in the 'test/' directory.
# We add the 'src/' directory (which is one level up) to the Python path
# to allow for the import of MCDCAnalyzer.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Now we can import the analyzer
try:
    import MCDCAnalyzer
except ImportError:
    print(f"Error: Could not import MCDCAnalyzer from {SRC_DIR}", file=sys.stderr)
    print("Please ensure MCDCAnalyzer.py is in the 'src' directory.", file=sys.stderr)
    sys.exit(1)

# --- Test Configuration ---
# Define the directory containing the test programs, metrics, and traces
TESTPROGRAMS_DIR = os.path.join(SCRIPT_DIR, 'testprograms')

# Create the list of test cases dynamically
TEST_CASE_NAMES = ["palindrome_partitioning", "markov_chain", "manhattan_distance"]
TEST_CASES = []

for name in TEST_CASE_NAMES:
    source_file = os.path.join(TESTPROGRAMS_DIR, f"{name}.py")
    metrics_file = os.path.join(TESTPROGRAMS_DIR, f"{name}_metrics.json")
    # Use the pre-filtered trace file
    trace_file = os.path.join(TESTPROGRAMS_DIR, f"filtered_{name}_trace.txt")
    
    # Check if all required files exist before adding the test case
    if all(os.path.exists(f) for f in [source_file, metrics_file, trace_file]):
        TEST_CASES.append((source_file, metrics_file, trace_file))
    else:
        print(f"Warning: Skipping test case '{name}' - missing one or more files in {TESTPROGRAMS_DIR}", file=sys.stderr)
        if not os.path.exists(source_file): print(f"  Missing: {source_file}")
        if not os.path.exists(metrics_file): print(f"  Missing: {metrics_file}")
        if not os.path.exists(trace_file): print(f"  Missing: {trace_file}")

# --- Fixture Setup ---

@pytest.fixture(scope="module", params=TEST_CASES, ids=[os.path.basename(item[0]) for item in TEST_CASES])
def analysis_results(request):
    """
    This fixture is the "setup function".
    It runs once for each program in TEST_CASES.
    It reads the pre-filtered trace file and runs the MCDC analysis
    to provide the necessary inputs for the evaluation functions.
    """
    source_filepath, metrics_filepath, filtered_trace_filepath = request.param

    # 1. --- Setup Logger ---
    # With no logging output.
    logger = logging.getLogger(f"test_analyzer_{os.path.basename(source_filepath)}")
    logger.setLevel(logging.CRITICAL)
    if not logger.hasHandlers():
        logger.addHandler(logging.NullHandler())

    # 2. --- Load the target source module ---
    # This is still required to populate the 'safe_functions_whitelist' 
    # for the condition evaluation step.
    module_name = os.path.splitext(os.path.basename(source_filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, source_filepath)
    target_module = importlib.util.module_from_spec(spec)
    
    # Add module's directory to path for correct imports, then remove it
    sys.path.insert(0, os.path.dirname(source_filepath))
    spec.loader.exec_module(target_module)
    sys.path.pop(0)

    # 3. --- Run MCDCAnalyzer Core Functions ---
    
    # Parse source code
    conditions = MCDCAnalyzer.parse_conditions(source_filepath)
    statements = MCDCAnalyzer.parse_statements(source_filepath)
    functions_in_file = MCDCAnalyzer.get_functions(source_filepath)
    total_branch_count = MCDCAnalyzer.count_branches(source_filepath, methods=functions_in_file)
    
    # Get trace runs from the *pre-filtered* file
    runs = MCDCAnalyzer.get_traces(filtered_trace_filepath, logger)
    
    # Analyze branch coverage
    branch_coverage_result = MCDCAnalyzer.analyze_trace_files_branch_coverage(conditions, runs, logger)
    
    # Prepare safe functions for evaluation
    safe_functions_whitelist = MCDCAnalyzer.SAFE_BUILTINS.copy()
    for func_name in functions_in_file:
        if hasattr(target_module, func_name):
            safe_functions_whitelist[func_name] = getattr(target_module, func_name)

    # Analyze condition/decision/MCDC coverage
    # use_builtin_eval=False for safety, as in Main.py
    condition_map, statement_map, _, decisions = \
        MCDCAnalyzer.analyze_trace_files_condition_coverage(
            conditions, statements, runs, safe_functions_whitelist, logger, use_builtin_eval=False
        )

    # 5. --- Load Expected Results from JSON ---
    with open(metrics_filepath, 'r') as f:
        expected_metrics = json.load(f)

    # 6. --- Clean up module ---
    # This is important to avoid conflicts between test runs
    if module_name in sys.modules:
        del sys.modules[module_name]

    # 7. --- Yield all results to the tests ---
    yield {
        "logger": logger,
        "branch_coverage_input": branch_coverage_result,
        "total_branch_count_input": total_branch_count,
        "decisions_input": decisions,
        "condition_map_input": condition_map,
        "statement_map_input": statement_map,
        "expected_metrics": expected_metrics
    }

# --- Pytest Tests ---

def test_branch_coverage(analysis_results):
    """
    Tests if the calculated branch coverage matches the JSON file.
    """
    # As requested, call the evaluation function inside the test.
    # We pass io.StringIO() as the file to prevent printing to console.
    calculated_metrics = MCDCAnalyzer.evaluate_branch_coverage(
        coverage=analysis_results["branch_coverage_input"],
        total_branch_count=analysis_results["total_branch_count_input"],
        logger=analysis_results["logger"],
        file=io.StringIO()
    )
    
    # Get expected metrics [covered, total] and convert to tuple
    expected_metrics = tuple(analysis_results["expected_metrics"]["branch"])
    
    assert calculated_metrics == expected_metrics

def test_decision_coverage(analysis_results):
    """
    Tests if the calculated decision coverage matches the JSON file.
    """
    calculated_metrics = MCDCAnalyzer.evaluate_decision_coverage(
        decisions=analysis_results["decisions_input"],
        logger=analysis_results["logger"],
        file=io.StringIO()
    )
    
    expected_metrics = tuple(analysis_results["expected_metrics"]["decision"])
    
    assert calculated_metrics == expected_metrics

def test_condition_coverage(analysis_results):
    """
    Tests if the calculated condition coverage matches the JSON file.
    """
    calculated_metrics = MCDCAnalyzer.evaluate_condition_coverage(
        condition_map=analysis_results["condition_map_input"],
        logger=analysis_results["logger"],
        file=io.StringIO()
    )
    
    expected_metrics = tuple(analysis_results["expected_metrics"]["condition"])
    
    assert calculated_metrics == expected_metrics

def test_mcdc_coverage(analysis_results):
    """
    Tests if the calculated MC/DC coverage matches the JSON file.
    """
    calculated_metrics = MCDCAnalyzer.evaluate_mcdc_coverage(
        condition_map=analysis_results["condition_map_input"],
        statement_map=analysis_results["statement_map_input"],
        logger=analysis_results["logger"],
        file=io.StringIO()
    )
    
    expected_metrics = tuple(analysis_results["expected_metrics"]["mcdc"])
    
    assert calculated_metrics == expected_metrics