#TraceGenerator.py
import doctest
import inspect
import sys
from Tracing import tracer

def generate_trace(target_module, trace_file_path: str, max_bytes: int | None = None):
    """
    Automatically discovers and runs all doctests in a target module,
    generating a single, ordered trace file for all test executions.
    This version directly modifies (monkey-patches) the target module to ensure tracing.
    """
    if max_bytes:
        print(f"  - Generating trace for module: {target_module.__name__} (Size limit: {max_bytes / (1024*1024):.2f} MB)")
    else:
        print(f"  - Generating trace for module: {target_module.__name__}")
    
    total_failures = 0
    total_attempts = 0

    with open(trace_file_path, "w") as trace_file_handle:
        # Create the tracer decorator with the file handle and size limit.
        tracer_decorator = tracer(trace_file_handle, max_bytes=max_bytes)
        
        objects_to_test = []

        # Part A: Discover and wrap standalone functions
        for name, func in inspect.getmembers(target_module, inspect.isfunction):
            if func.__module__ == target_module.__name__:
                # Monkey-patch the function in the module itself.
                setattr(target_module, name, tracer_decorator(func))
                if func.__doc__:
                    objects_to_test.append((name, func))

        # Part B: Discover and wrap classes and their methods
        for class_name, cls in inspect.getmembers(target_module, inspect.isclass):
            if cls.__module__ == target_module.__name__:
                # A list to hold the original methods before patching
                original_methods = []
                for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                    original_methods.append((method_name, method))
                    traced_method = tracer_decorator(method)
                    # Patch the method directly on the class object.
                    setattr(cls, method_name, traced_method)
                
                # Check for doctests on the class itself.
                if cls.__doc__:
                    objects_to_test.append((class_name, cls))
                
                # Also, check if any of the original methods have doctests.
                for method_name, method in original_methods:
                    if method.__doc__:
                        qualified_name = f"{class_name}.{method_name}"
                        objects_to_test.append((qualified_name, method))

        if not objects_to_test:
            print("  - No functions, methods, or classes with doctests found.")
            return

        # Now run all the collected doctests using the module's own dictionary as globals.
        parser = doctest.DocTestParser()
        runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)

        # The globals for the doctest will be the module's own dictionary,
        # which now contains the wrapped functions and patched classes.
        test_globals = target_module.__dict__

        for test_name, original_object in objects_to_test:
            try:
                # NOTE: We pass the original object to find the docstring, but the
                # runner executes code in the context of test_globals where everything is patched.
                test = parser.get_doctest(
                    string=original_object.__doc__,
                    globs=test_globals,
                    name=test_name,
                    filename=inspect.getfile(original_object),
                    lineno=inspect.getsourcelines(original_object)[1]
                )
            except (ValueError, TypeError):
                continue

            # Run the test if it contains any examples.
            if test and test.examples:
                failures, attempts = runner.run(test)
                if attempts > 0:
                    # Update the summary counters
                    total_failures += failures
                    total_attempts += attempts
    
    # The main script handles un-doing the patches by deleting the module from sys.modules.

    print(f"  - Doctest summary: {total_attempts} tests run, {total_failures} failures.")