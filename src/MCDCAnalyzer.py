# MCDCAnalyzer.py (with advanced logging)
import ast
import os
import sys
import json
import operator as op
import importlib.util
from datetime import datetime
import logging
from pathlib import Path

# Supported operators for safe evaluation
OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.BitXor: op.xor, ast.USub: op.neg, ast.Eq: op.eq,
    ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge,
    ast.And: lambda a, b: a and b, ast.Or: lambda a, b: a or b, ast.Not: op.not_,
    ast.Mod: op.mod, ast.BitAnd: op.and_, ast.In: lambda e, c: e in c,
    ast.NotIn: lambda e, c: e not in c,
}

# Evaluator that safely evaluates expressions using AST
class SafeEvaluator:
    def __init__(self, functions: dict, context: dict, logger: logging.Logger):
        self.functions = functions
        self.namespace = {**functions, **context}
        self.logger = logger

    def eval(self, expression: str):
        try:
            tree = ast.parse(expression, mode='eval')
            return self._visit(tree.body)
        except Exception as e:
            self.logger.exception(f"Failed to evaluate expression '{expression}': {e}")
            raise type(e)(f"Failed to evaluate expression '{expression}': {e}") from e

    def _visit(self, node):
        node_type = type(node)
        
        if node_type is ast.Constant:
            return node.value
        
        elif node_type is ast.Name:
            try:
                return self.namespace[node.id]
            except KeyError:
                raise NameError(f"Name '{node.id}' is not defined in the current context.")

        elif node_type is ast.Attribute:
            obj = self._visit(node.value)
            attr_name = node.attr
            if isinstance(obj, dict):
                try:
                    return obj[attr_name]
                except KeyError:
                    raise NameError(f"Attribute '{attr_name}' not found in traced object dictionary.")
            else:
                try:
                    return getattr(obj, attr_name)
                except AttributeError:
                    raise NameError(f"Attribute '{attr_name}' not found on object.")
        
        elif node_type is ast.BinOp:
            left = self._visit(node.left)
            right = self._visit(node.right)
            return OPERATORS[type(node.op)](left, right)
        
        elif node_type is ast.UnaryOp:
            operand = self._visit(node.operand)
            return OPERATORS[type(node.op)](operand)
            
        elif node_type is ast.Compare:
            left = self._visit(node.left)
            for i, op_node in enumerate(node.ops):
                right = self._visit(node.comparators[i])
                if not OPERATORS[type(op_node)](left, right):
                    return False
                left = right
            return True

        elif node_type is ast.BoolOp:
            op_type = type(node.op)
            for value_node in node.values:
                result = self._visit(value_node)
                if (op_type is ast.And and not result) or (op_type is ast.Or and result):
                    return result
            return result
            
        elif node_type is ast.Call:
            func_name = node.func.id
            if func_name not in self.functions:
                raise NameError(f"Function '{func_name}' is not a whitelisted callable.")
            args = [self._visit(arg) for arg in node.args]
            kwargs = {kw.arg: self._visit(kw.value) for kw in node.keywords}
            return self.functions[func_name](*args, **kwargs)

        elif node_type is ast.List: return [self._visit(e) for e in node.elts]
        elif node_type is ast.Tuple: return tuple(self._visit(e) for e in node.elts)
        elif node_type is ast.Subscript: return self._visit(node.value)[self._visit(node.slice)]
        else: raise TypeError(f"Unsupported node type: {node_type.__name__}")

# Whitelisted built-in functions for safe evaluation
SAFE_BUILTINS = {
    "isinstance": isinstance, "issubclass": issubclass, "type": type, "int": int, "str": str, 
    "float": float, "bool": bool, "list": list, "tuple": tuple, "len": len, "all": all, "any": any,
    "sorted": sorted, "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    "True": True, "False": False, "None": None
}

# AST Visitors for parsing source code

class ConditionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.conditions = []
    def visit_If(self, node):
        true_lines = {n.lineno for n in node.body if hasattr(n, 'lineno')}
        false_lines = {n.lineno for n in node.orelse if hasattr(n, 'lineno')}
        self.conditions.append({'lineno': node.lineno, 'true_lines': true_lines, 'false_lines': false_lines})
        self.generic_visit(node)
    def visit_While(self, node):
        true_lines = {n.lineno for n in node.body if hasattr(n, 'lineno')}
        false_lines = {n.lineno for n in node.orelse if hasattr(n, 'lineno')}
        self.conditions.append({'lineno': node.lineno, 'true_lines': true_lines, 'false_lines': false_lines})
        self.generic_visit(node)

class StatementVisitor(ast.NodeVisitor):
    def __init__(self):
        self.statements = {}
    def visit_If(self, node):
        self.statements[node.lineno] = ast.unparse(node.test).strip()
        self.generic_visit(node)
    def visit_While(self, node):
        self.statements[node.lineno] = ast.unparse(node.test).strip()
        self.generic_visit(node)

class SourceLineVisitor(ast.NodeVisitor):
    def __init__(self):
        self.lines = set()
    def generic_visit(self, node):
        if hasattr(node, 'lineno'): self.lines.add(node.lineno)
        super().generic_visit(node)

class BranchCounter(ast.NodeVisitor):
    def __init__(self):
        self.if_branch_counts = []
    def visit_If(self, node):
        #self.if_branch_counts.append((node.lineno, 2 if node.orelse else 1))
        #self.if_branch_counts.append((node.lineno, 1 + (1 if getattr(node, 'orelse', None) else 0))) 
        # previous solution:
        self.if_branch_counts.append((node.lineno, 2)) 
        self.generic_visit(node)
    def visit_While(self, node):
        #self.if_branch_counts.append((node.lineno, 2 if node.orelse else 1))
        self.if_branch_counts.append((node.lineno, 2)) #A while loop always has two branches.
        self.generic_visit(node)

# Parsing and analysis functions

def parse_conditions(filename):
    with open(filename, "r", encoding='utf-8') as file:
        visitor = ConditionVisitor()
        visitor.visit(ast.parse(file.read(), filename=filename))
        return visitor.conditions

def parse_statements(filename):
    with open(filename, "r", encoding='utf-8') as file:
        visitor = StatementVisitor()
        visitor.visit(ast.parse(file.read(), filename=filename))
        return visitor.statements

def parse_source_lines(filename):
    with open(filename, "r", encoding='utf-8') as file:
        visitor = SourceLineVisitor()
        visitor.visit(ast.parse(file.read(), filename=filename))
        return visitor.lines

# Count branches in a specific method
def count_if_branches(filename, method_name):
    with open(filename, "r", encoding='utf-8') as source:
        tree = ast.parse(source.read(), filename=filename)
    function_root = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == method_name), None)
    if function_root is None: return []
    visitor = BranchCounter()
    visitor.visit(function_root)
    return visitor.if_branch_counts

# Get all function names in the file
def get_functions(filepath):
    with open(filepath, "r", encoding='utf-8') as source:
        tree = ast.parse(source.read())
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

# Extract traces from the trace file
def get_traces(trace_file, logger):
    runs = {}
    logger.debug(f"Reading traces from {trace_file}")
    with open(trace_file, "r", encoding='utf-8') as f:
        current_trace_id = None
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Trace"):
                current_trace_id = stripped.split(' ', 1)[0]
                runs[current_trace_id] = []
            elif stripped.startswith("Line") and current_trace_id:
                runs[current_trace_id].append(stripped)
    logger.info(f"Found {len(runs)} unique trace runs.")
    return runs

# Filter and clean trace lines from the trace file
def filter_and_clean_trace_lines(input_file, output_file, source_lines, logger):
    logger.debug(f"Filtering trace lines from {input_file} to {output_file}")
    lines_written = 0
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            #logger.debug(f"Processing line: {line}")
            stripped = line.strip()
            #logger.debug(f"Stripped line: {stripped}")
            if not stripped: continue
            if stripped.startswith("Trace"):
                outfile.write(stripped + "\n")
                lines_written += 1
                continue
            if stripped.startswith("Line"):
                try:
                    line_num = int(stripped.split(' ', 2)[1])
                    if line_num in source_lines:
                        outfile.write(stripped + "\n")
                        lines_written += 1
                except (IndexError, ValueError):
                    logger.warning(f"Ignoring malformed trace line: {stripped}")
                    continue
    logger.debug(f"Wrote {lines_written} lines to filtered trace file.")

# Analyze trace files for branch coverage for all conditions
def analyze_trace_files_branch_coverage(conditions, runs, logger):
    logger.debug("Analyzing trace files for branch coverage")
    coverage = {c['lineno']: {'true': False, 'false': False} for c in conditions}

    for run_id, run_lines in runs.items():
        executed_in_run = {int(line.split()[1]) for line in run_lines if line.startswith("Line")}
        
        for c in conditions:
            # If the decision line itself was never executed, skip it
            if c['lineno'] not in executed_in_run:
                continue

            # 1. Check for TRUE branch coverage (same as before)
            # This is covered if any line inside the loop body was executed.
            if not coverage[c['lineno']]['true'] and any(l in executed_in_run for l in c['true_lines']):
                coverage[c['lineno']]['true'] = True
                logger.debug(f"Run {run_id}: Covered TRUE branch for decision at line {c['lineno']}")

            # 2. Check for FALSE branch coverage 
            # Check the original orelse condition
            if not coverage[c['lineno']]['false'] and any(l in executed_in_run for l in c['false_lines']):
                     coverage[c['lineno']]['false'] = True
                     logger.debug(f"Run {run_id}: Covered FALSE branch for decision at line {c['lineno']} (orelse block)")
            elif not coverage[c['lineno']]['false']:
                # The false branch is covered if execution continued to a line *after* the decision,
                # that isn't part of another decision's body we are tracking.
                # This robustly handles loop exits.
                
                # Find the maximum line number within the true branch (the loop body)
                max_true_line = max(c['true_lines']) if c['true_lines'] else c['lineno']
                
                # See if any line *after* the loop body was executed
                has_executed_after_loop = any(l > max_true_line for l in executed_in_run)
                
                if has_executed_after_loop:
                    coverage[c['lineno']]['false'] = True
                    logger.debug(f"Run {run_id}: Covered FALSE branch for decision at line {c['lineno']} (inferred from loop exit)")
                

    logger.debug(f"Final branch coverage result: {coverage}")
    return coverage

# Extracting conditions from statements for adding to condition map
def extract_conditions_from_statement(id, lineno, stmt_from_ast, conditions_by_line_number):
    atomic_conditions = []
    class ConditionExtractor(ast.NodeVisitor):
        def visit_BoolOp(self, node):
            for value in node.values: self.visit(value)
        def generic_visit(self, node):
            if not isinstance(node, ast.BoolOp): atomic_conditions.append(ast.unparse(node).strip())
    try:
        visitor = ConditionExtractor()
        visitor.visit(ast.parse(stmt_from_ast, mode='eval').body)
    except SyntaxError:
        atomic_conditions.append(stmt_from_ast)
    if lineno not in conditions_by_line_number: conditions_by_line_number[lineno] = {}
    for condition in atomic_conditions:
        id += 1
        conditions_by_line_number[lineno][id] = condition
    return conditions_by_line_number, id

# Creating condition dictionary for a statement line number
def create_condition_dict(statement_lineno, condition_map, lineno, conditions_by_line_number):
    for _, cond in conditions_by_line_number[lineno].items():
        if cond.strip() not in condition_map[statement_lineno]:
            condition_map[statement_lineno][cond.strip()] = []
    return condition_map

# Recording coverage for a specific condition using safe evaluation
def record_coverage_for_condition(booleans, statement_lineno, condition_map, condition, local_variables, safe_functions, logger, use_builtin_eval: bool):
    evaluation = None
    try:
        if use_builtin_eval:
            # When using builtin eval, the context is just the local variables from the trace.
            # The 'safe_functions' list is intentionally not used, per the requirement.
            # The standard Python built-ins will be available by default.
            logger.debug(f"Evaluating condition with built-in eval: {condition}")
            evaluation = bool(eval(condition, local_variables))
        else:
            evaluator = SafeEvaluator(functions=safe_functions, context=local_variables, logger=logger)
            logger.debug(f"Evaluating condition with SafeEvaluator: {condition}")
            evaluation = bool(evaluator.eval(condition))
    except Exception as e:
        logger.error(f"Could not evaluate condition '{condition}' at line {statement_lineno}. Reason: {e}")
    booleans.append(evaluation)
    if evaluation is not None:
        condition_map[statement_lineno][condition].append(evaluation)
    return booleans, condition_map

# Adding statement evaluation to the list of booleans for decision outcome
def add_statement_evaluation_to_list_of_booleans(booleans, local_variables, stmt_from_ast, safe_functions, logger, use_builtin_eval: bool):
    evaluation = None
    try:
        if use_builtin_eval:
            # When using builtin eval, the context is just the local variables from the trace.
            # The 'safe_functions' list is intentionally not used, per the requirement.
            # The standard Python built-ins will be available by default.
            logger.debug(f"Evaluating statement with built-in eval: {stmt_from_ast}")
            evaluation = bool(eval(stmt_from_ast, local_variables))
        else:
            evaluator = SafeEvaluator(functions=safe_functions, context=local_variables, logger=logger)
            logger.debug(f"Evaluating statement with SafeEvaluator: {stmt_from_ast}")
            evaluation = bool(evaluator.eval(stmt_from_ast))
    except Exception as e:
        logger.error(f"Could not evaluate statement '{stmt_from_ast}'. Reason: {e}")
    booleans.append(evaluation)
    return booleans

# Procedure for condition coverage analysis per condition with all necessary updates and steps 
def condition_coverage_procedure(statement_map, tuples, statement_lineno, condition_map, statements, condition, vars_from_trace, id, conditions_by_line_number, decisions, safe_functions, logger, use_builtin_eval: bool):
    #logger.debug(f"Processing procedure with condition at line {condition['lineno']} for statement line {statement_lineno}")
    lineno, stmt_from_ast = condition['lineno'], statements[condition['lineno']]
    local_vars = vars_from_trace[lineno]
    # Initialize maps if not already present
    if statement_lineno not in decisions: decisions[statement_lineno] = []
    if statement_lineno not in statement_map: statement_map[statement_lineno] = []
    # Extract atomic conditions from the statement and update the condition map
    conditions_by_line_number, id = extract_conditions_from_statement(0, lineno, stmt_from_ast, {})
    condition_map = create_condition_dict(statement_lineno, condition_map, lineno, conditions_by_line_number)
    booleans = []
    # Record conditions of the statement line numbers and record their evaluations by connecting AST nodes to trace variables and line numbers
    for _, cond in conditions_by_line_number[lineno].items():
        booleans, condition_map = record_coverage_for_condition(booleans, statement_lineno, condition_map, cond, local_vars, safe_functions, logger, use_builtin_eval) 
    # Finally, evaluate the overall decision outcome and add it to the list of booleans
    booleans = add_statement_evaluation_to_list_of_booleans(booleans, local_vars, stmt_from_ast, safe_functions, logger, use_builtin_eval) 
    # Record the decision outcome (last boolean) for decision coverage
    decisions[statement_lineno].append(booleans[-1])
    # Add the full tuple of condition evaluations + decision outcome to the tuples list and statement map
    tuples.append(tuple(booleans))
    # Only add non-None tuples to the statement map
    statement_map[statement_lineno].append(tuple(booleans))
    #logger.debug(f"Updated statement_map with procedure for line {statement_lineno}: {statement_map[statement_lineno]}")
    return statement_map, tuples, condition_map, id, conditions_by_line_number, decisions

# Analyze trace files for condition/decision/MC-DC coverage
def analyze_trace_files_condition_coverage(conditions, statements, runs, safe_functions, logger, use_builtin_eval: bool):
    logger.debug("Analyzing trace files for condition/decision/MC-DC coverage.")
    # Initialize maps to store coverage information
    condition_map, statement_map, tuples, decisions = {}, {}, [], {}
    # Process each unique run only once
    unique_runs = {tuple(run) for run in runs.values()}
    # Define separator used in trace lines
    separator = "|||---|||"
    
    run_count = 0
    for run in unique_runs:
        run_count += 1
        logger.debug(f"Processing run {run_count}/{len(unique_runs)}")
        #logger.debug(f"Run content: {run}") WARNING: This may produce very large logs.
        # Parse the trace lines to extract variable states by line number
        vars_by_line = {}
        for line in run:
            logger.debug(f"Processing trace line: {line}")
            # Parse lines starting with "Line" to extract line number and variable states
            if line.startswith("Line"):
                parts = line.strip().split(maxsplit=2)
                if len(parts) >= 3:
                    try:
                        lineno = int(parts[1])
                        json_parts = parts[2].split(separator)
                        #logger.debug(f"Parsed line {lineno} with vars: {json_parts}")
                        if len(json_parts) == 2:
                            local_vars = json.loads(json_parts[0])
                            global_vars = json.loads(json_parts[1])
                            #logger.debug(f"Extracted local vars for line {lineno}: {local_vars}")
                            #logger.debug(f"Extracted global vars for line {lineno}: {global_vars}")
                            vars_by_line.setdefault(lineno, []).append({**global_vars, **local_vars})
                            #logger.debug(f"Combined vars for line {lineno}: {vars_by_line[lineno]}")
                    except (ValueError, json.JSONDecodeError) as e:
                        logger.error(f"Failed to parse trace line: '{line}'. Reason: {e}")
                        continue
        # For each condition, evaluate it using the variable states from the trace
        for condition in conditions:
            #logger.debug(f"Evaluating condition: {condition}")
            # Only process if we have variable states for the condition's line number
            if condition['lineno'] in vars_by_line:
                # Ensure the condition map has an entry for this line number
                condition_map.setdefault(condition['lineno'], {})
                # Evaluate the condition for each set of variable states recorded at that line number
                for single_vars in vars_by_line[condition['lineno']]:
                    #logger.debug(f"Evaluating condition with vars: {single_vars}")
                    temp_vars_map = {condition['lineno']: single_vars}
                    # Evaluate the condition using the current variable mapping
                    statement_map, tuples, condition_map, _, _, decisions = \
                        condition_coverage_procedure(statement_map, tuples, condition['lineno'], condition_map, statements, condition, temp_vars_map, 0, {}, decisions, safe_functions, logger, use_builtin_eval) 
    logger.info("Finished evaluating conditions from trace runs.")
    return condition_map, statement_map, tuples, decisions

# Calculate MC/DC pairs for each condition
def calculate_mcdc_pairs(ids, trues, falses):
    mcdc_coverage_result, mcdc_pairs_result = [False] * ids, [[] for _ in range(ids)]
    for i in range(ids):
        # Find pairs of tuples that differ only in the i-th condition and have different outcomes
        for tuple_true in trues[i]:
            # Skip if we already found a pair for this condition
            if mcdc_coverage_result[i]: break
            # Look for a matching false tuple
            for tuple_false in falses[i]:
                # Check if they differ only in the i-th condition and have different outcomes
                if len(tuple_true) == len(tuple_false) and tuple_true[i] != tuple_false[i] and tuple_true[-1] != tuple_false[-1] and \
                   all(tuple_true[j] == tuple_false[j] for j in range(len(tuple_true)) if j not in {i, len(tuple_true) - 1}):
                    # Found a valid MC/DC pair
                    mcdc_pairs_result[i].append((tuple_true, tuple_false))
                    mcdc_coverage_result[i] = True
                    break
    return mcdc_coverage_result, mcdc_pairs_result, all(mcdc_coverage_result)

# Count total branches in all methods of the file
def count_branches(filename, methods):
    total_branch_count = 0
    for method_name in methods:
        branch_count_list = count_if_branches(filename, method_name)
        total_branch_count += sum(count for _, count in branch_count_list)
    return total_branch_count

# Evaluation functions for coverage metrics

def evaluate_branch_coverage(coverage, total_branch_count, logger, file=sys.stdout):
    logger.info(f"Evaluating Branch Coverage...")
    # Count covered branches
    covered_branches = sum(1 for branches in coverage.values() if branches['true']) + \
                       sum(1 for branches in coverage.values() if branches['false'])
    coverage_percent = (covered_branches / total_branch_count * 100) if total_branch_count > 0 else 0
    logger.info(f"Branch Coverage: {covered_branches}/{total_branch_count} ({coverage_percent:.2f}%)")
    # Print detailed results
    print("\n--- Branch Coverage Results ---", file=file)
    print(f"Covered Branches: {covered_branches} / {total_branch_count}", file=file)
    print(f"Branch Coverage: {coverage_percent:.2f}%\n", file=file)
    return covered_branches, total_branch_count

def evaluate_decision_coverage(decisions, logger, file=sys.stdout):
    logger.info(f"Evaluating Decision Coverage...")
    # Count covered decisions
    covered_decisions = sum(1 for b in decisions.values() if True in b and False in b)
    total_decisions = len(decisions)
    coverage_percent = (covered_decisions / total_decisions * 100) if total_decisions > 0 else 0
    logger.info(f"Decision Coverage: {covered_decisions}/{total_decisions} ({coverage_percent:.2f}%)")
    # Print detailed results
    print("\n--- Decision Coverage Results ---", file=file)
    for lineno, booleans in sorted(decisions.items()):
        status = "covered" if True in booleans and False in booleans else "NOT covered"
        print(f"Decision at line {lineno} is {status}.", file=file)
        logger.debug(f"Decision at line {lineno} is {status}. Outcomes observed: {booleans}")
    print(f"\nTotal Decisions Covered: {covered_decisions} / {total_decisions}", file=file)
    print(f"Decision Coverage: {coverage_percent:.2f}%\n", file=file)
    return covered_decisions, total_decisions
    
def evaluate_condition_coverage(condition_map, logger, file=sys.stdout):
    logger.info(f"Evaluating Condition Coverage...")
    covered, total_conditions = 0, 0
    print("\n--- Condition Coverage Results ---", file=file)
    # Print detailed results
    for line_num, condition_dict in sorted(condition_map.items()):
        print(f"Decision at line {line_num}:", file=file)
        for condition_name, bool_list in condition_dict.items():
            total_conditions += 1
            status = "covered" if True in bool_list and False in bool_list else "NOT covered"
            print(f"  - Condition '{condition_name}' is {status}.", file=file)
            logger.debug(f"Line {line_num} - Condition '{condition_name}' is {status}. Outcomes: {bool_list}")
            if status == "covered": covered += 1
    coverage_percent = (covered / total_conditions * 100) if total_conditions > 0 else 0
    logger.info(f"Condition Coverage: {covered}/{total_conditions} ({coverage_percent:.2f}%)")
    print(f"\nTotal Conditions Covered: {covered} / {total_conditions}", file=file)
    print(f"Condition Coverage: {coverage_percent:.2f}%\n", file=file)
    return covered, total_conditions

def evaluate_mcdc_coverage(condition_map, statement_map, logger, file=sys.stdout):
    logger.info(f"Evaluating MC/DC Coverage...")
    mcdc_coverage_count, total_decisions = 0, 0
    # Print detailed results
    print("\n--- MC/DC Coverage Results ---", file=file)
    # Evaluate MC/DC for each decision
    for line_num, condition_dict in sorted(condition_map.items()):
        total_decisions += 1
        id_by_line_number = len(condition_dict)
        if id_by_line_number == 0: continue
        # Prepare true/false candidate maps for each condition
        mcdc_candidates_true = {i: [] for i in range(id_by_line_number)}
        mcdc_candidates_false = {i: [] for i in range(id_by_line_number)}
        # Populate candidate maps from the statement map
        for t in statement_map.get(line_num, []):
            # Skip tuples with None values or incorrect length
            if None in t: continue
            if len(t) != id_by_line_number + 1: continue
            for i in range(id_by_line_number):
                # Append to true/false candidates based on the condition's evaluation
                (mcdc_candidates_true if t[-1] else mcdc_candidates_false)[i].append(t)
        # Calculate MC/DC pairs and coverage status
        _, _, covered = calculate_mcdc_pairs(id_by_line_number, mcdc_candidates_true, mcdc_candidates_false)
        # Print results for this decision
        status = "PASSED" if covered else "FAILED"
        print(f"MC/DC for decision at line {line_num}: {status}", file=file)
        logger.debug(f"MC/DC for decision at line {line_num}: {status}")
        # Update coverage counts
        if covered: mcdc_coverage_count += 1
        # If not covered, print which conditions failed
        elif id_by_line_number > 0:
            # Recalculate to get detailed coverage results
            mcdc_coverage_result, _, _ = calculate_mcdc_pairs(id_by_line_number, mcdc_candidates_true, mcdc_candidates_false)
            # Map condition indices to their names
            conditions = list(condition_dict.keys())
            # Print uncovered conditions
            if not conditions or not mcdc_coverage_result: continue
            # There should be a one-to-one mapping between conditions and coverage results
            for i, is_cond_covered in enumerate(mcdc_coverage_result):
                if not is_cond_covered:
                    msg = f"  - Independence not shown for condition '{conditions[i]}'"
                    print(msg, file=file)
                    logger.debug(f"Line {line_num}: " + msg)

    # Final summary
    coverage_percent = (mcdc_coverage_count / total_decisions * 100) if total_decisions > 0 else 0
    logger.info(f"MC/DC Coverage: {mcdc_coverage_count}/{total_decisions} ({coverage_percent:.2f}%)")
    print(f"\nOverall MC/DC Decisions Met: {mcdc_coverage_count} / {total_decisions}", file=file)
    print(f"MC/DC Coverage: {coverage_percent:.2f}%", file=file)
    return mcdc_coverage_count, total_decisions

# Main analysis function with advanced logging and error handling
def analyze(source_filepath: str, trace_filepath: str, report_filepath: str, json_report_filepath: str, log_root_path: str, log_subfolder_name: str, use_builtin_eval: bool = False): 
    # Set up temporary logger
    program_name = os.path.splitext(os.path.basename(source_filepath))[0]
    logger = logging.getLogger(f"analyzer.{program_name}")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create log directory
    temp_log_dir = Path(log_root_path) / "error logs" / log_subfolder_name / ".tmp"
    temp_log_dir.mkdir(parents=True, exist_ok=True)
    temp_log_filepath = temp_log_dir / f"{program_name}.log.tmp"

    fh = logging.FileHandler(temp_log_filepath, mode='w', encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(fh)

    logger.info(f"--- Starting analysis for {source_filepath} ---")
    logger.info(f"Trace file: {trace_filepath}")
    logger.info(f"Report file: {report_filepath}")
    
    # Initialize metrics dictionary
    metrics = {'branch': (0, 0), 'decision': (0, 0), 'condition': (0, 0), 'mcdc': (0, 0)}
    
    try:
        with open(report_filepath, 'w', encoding='utf-8') as report_file:
            # Write header to report file
            report_file.write(f"Analysis Report for: {os.path.basename(source_filepath)}\n")
            report_file.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_file.write("="*40 + "\n")

            # Parse source code structure
            logger.info("Parsing source code structure...")
            source_lines = parse_source_lines(source_filepath)
            conditions = parse_conditions(source_filepath)
            statements = parse_statements(source_filepath)
            functions_in_file = get_functions(source_filepath)
            logger.info(f"Found {len(conditions)} decisions and {len(functions_in_file)} functions.")

            # Copy safe built-ins and dynamically load target module functions
            safe_functions_whitelist = SAFE_BUILTINS.copy()
            try:
                module_name = os.path.splitext(os.path.basename(source_filepath))[0]
                spec = importlib.util.spec_from_file_location(module_name, source_filepath)
                target_module = importlib.util.module_from_spec(spec)
                sys.path.insert(0, os.path.dirname(source_filepath))
                spec.loader.exec_module(target_module)
                sys.path.pop(0)
                for func_name in functions_in_file:
                    if hasattr(target_module, func_name): safe_functions_whitelist[func_name] = getattr(target_module, func_name)
                logger.info("Successfully loaded target module for function evaluation.")
            except Exception as e:
                logger.error(f"Could not dynamically load module '{source_filepath}'. Reason: {e}")
            
            # Count total branches in the source file
            total_branch_count = count_branches(source_filepath, methods=functions_in_file)
            logger.info(f"Calculated total potential branches: {total_branch_count}")

            # Process and prepare trace file for analysis
            logger.info("Processing trace file...")
            filtered_trace_file = os.path.join(os.path.dirname(trace_filepath), f"filtered_{os.path.basename(trace_filepath)}")
            filter_and_clean_trace_lines(trace_filepath, filtered_trace_file, source_lines, logger)
            
            # Extract traces
            runs = get_traces(filtered_trace_file, logger)
            
            # Analyze branch coverage
            logger.info("Analyzing branch coverage...")
            branch_coverage = analyze_trace_files_branch_coverage(conditions, runs, logger)
            metrics['branch'] = evaluate_branch_coverage(branch_coverage, total_branch_count, logger, file=report_file)
            
            # Analyze condition/decision/MC-DC coverage
            logger.info("Analyzing condition/decision/MC-DC coverage...")
            condition_map, statement_map, _, decisions = \
                analyze_trace_files_condition_coverage(conditions, statements, runs, safe_functions_whitelist, logger, use_builtin_eval)
            

            # Evaluate coverage metrics
            metrics['decision'] = evaluate_decision_coverage(decisions, logger, file=report_file)
            metrics['condition'] = evaluate_condition_coverage(condition_map, logger, file=report_file)
            metrics['mcdc'] = evaluate_mcdc_coverage(condition_map, statement_map, logger, file=report_file)

        # Write summary to report file
        with open(json_report_filepath, 'w', encoding='utf-8') as json_file:
            json.dump(metrics, json_file, indent=4)
        logger.info(f"Metrics saved to {json_report_filepath}")

    except Exception as e:
        logger.critical(f"A critical error occurred during analysis: {e}", exc_info=True)
    finally:
        # Calculate final results and move log file
        has_uncovered_items = any(val[0] < val[1] for val in metrics.values())
        has_covered_items = any(val[0] > 0 for val in metrics.values())

        # Determine status folder based on coverage results
        if not has_covered_items and any(val[1] > 0 for val in metrics.values()):
            status_folder = "failed programs" # All coverage is 0%
        elif not has_uncovered_items and has_covered_items:
            status_folder = "complete coverage" # All possible items are covered
        else:
            status_folder = "partly covered" # A mix, or a file with no coverable items

        # Calculate total coverage percentage for naming the log file
        total_covered = sum(val[0] for val in metrics.values())
        total_items = sum(val[1] for val in metrics.values())
        total_coverage_percent = (total_covered / total_items * 100) if total_items > 0 else 100.0

        # Move temporary log file to final location
        log_dir = Path(log_root_path) / "error logs" / log_subfolder_name / status_folder
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_name = f"{program_name}_coverage_{total_coverage_percent:.2f}%.log"
        final_log_filepath = log_dir / log_file_name

        # Final logging messages
        logger.info(f"Final coverage status: {status_folder.replace('_', ' ').title()}")
        logger.info(f"Analysis complete. Log will be moved to {final_log_filepath}")
        logger.info("--- End of analysis ---")

        # Clean up logger handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Move log file to final destination
        if final_log_filepath.exists():
            final_log_filepath.unlink()
        temp_log_filepath.rename(final_log_filepath)

    return metrics
