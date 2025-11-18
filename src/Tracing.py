#Tracing.py
import sys
import inspect
import json

def _expand_object(obj, seen=None):
    """
    Recursively expands a Python object into a structure that is safe for
    JSON serialization, handling circular references and non-serializable types.
    """
    if seen is None:
        seen = set()

    # If we have seen this exact object ID before in this path, it's a circular reference.
    if id(obj) in seen:
        return f"<CircularReference id={id(obj)}>"

    # Handle non-serializable types before recursion

    # Handle common iterators, generators, and sets by converting them to lists.
    # This checks for a '__next__' method to identify iterators/generators.
    if isinstance(obj, (set, range, map, filter)) or hasattr(obj, '__next__'):
        try:
            # Convert to list to capture the state. Pass a copy of 'seen' to each item.
            return [_expand_object(item, seen.copy()) for item in list(obj)]
        except Exception:
            return f"<Unserializable Iterator: {repr(obj)}>"

    # Handle functions, methods, and other callables.
    if callable(obj):
        return f"<Callable: {getattr(obj, '__name__', repr(obj))}>"

    # For all other objects, add them to the 'seen' set for this path.
    seen.add(id(obj))

    # Handle container types with recursion

    # For objects with attributes, expand their __dict__.
    if hasattr(obj, '__dict__'):
        try:
            attrs = {'__class__': obj.__class__.__name__}
            # Pass a copy of 'seen' to each attribute's expansion.
            for k, v in vars(obj).items():
                attrs[k] = _expand_object(v, seen.copy())
            return attrs
        except Exception:
            return repr(obj)
    
    # For lists and tuples, expand their items.
    elif isinstance(obj, (list, tuple)):
        # Pass a copy of 'seen' to each item's expansion. This is the key
        # fix for the incorrect circular reference detection.
        return [_expand_object(item, seen.copy()) for item in obj]
    
    # Otherwise, it's a primitive type (int, str, bool, etc.) that is JSON-safe.
    else:
        return obj

def trace_lines(frame, event, arg, file, target_filename, max_bytes: int | None, stop_flag: list[bool]):
    """Trace line execution in the target file."""

    # If the stop flag has been set, do nothing.
    if stop_flag[0]:
        return
    
    # Check file size limit before proceeding.
    if max_bytes is not None:
        file.flush()
        if file.tell() >= max_bytes:
            # Write a final message, set the flag, and stop.
            print(f"Trace file size limit of {max_bytes / (1024*1024):.2f} MB reached. Halting trace.", file=file)
            stop_flag[0] = True
            return

    co = frame.f_code
    if event != 'line' or co.co_filename != target_filename: return
    line_no = frame.f_lineno
    
    # Represent locals and globals using the robust _expand_object function.
    locals_repr = {k: _expand_object(v) for k, v in frame.f_locals.items()}
    globals_repr = {}
    for k, v in frame.f_globals.items():
        if not k.startswith('__') and not inspect.ismodule(v) and \
           not inspect.isclass(v) and not inspect.isfunction(v):
            globals_repr[k] = _expand_object(v)
            
    separator = "|||---|||"
    try:
        locals_json = json.dumps(locals_repr)
        globals_json = json.dumps(globals_repr)
        print(f"Line {line_no} {locals_json}{separator}{globals_json}", file=file)
    except TypeError as e:
        print(f"Line {line_no} Error: Could not serialize trace data to JSON: {e}", file=file)

def trace_calls(frame, event, arg, file, target_filename, max_bytes: int | None, stop_flag: list[bool]):
    """Trace function calls in the target file."""

    if event != 'call': return
    co = frame.f_code
    if co.co_filename != target_filename: return
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename
    return lambda f, e, a: trace_lines(f, e, a, file, target_filename, max_bytes, stop_flag)

def tracer(file, max_bytes: int | None = None):
    """Decorator to trace function calls and line executions, writing to a file."""

    counter = [0]
    stop_tracing_flag = [False] # A mutable flag to stop tracing when the file is full.

    # The actual decorator function
    def decorator(func):
        target_filename = func.__code__.co_filename
        def wrapper(*args, **kwargs):
            # Reset the stop flag for each new top-level call.
            stop_tracing_flag[0] = False
            
            # Check if the file is already full before starting a new trace.
            if max_bytes is not None:
                file.flush()
                if file.tell() >= max_bytes:
                    # Run the function without tracing if the file is full.
                    return func(*args, **kwargs)

            # Write a header for this function call
            current_count = counter[0]
            counter[0] += 1
            arg_strings = [repr(arg) for arg in args]
            kwarg_strings = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            all_args_str = " ".join(arg_strings + kwarg_strings)
            header = f"Trace{current_count} {all_args_str}"
            file.write(header + "\n")
            
            original_trace_function = sys.gettrace()
            try:
                # Pass the size limit and stop flag down to the trace functions.
                sys.settrace(lambda *f_args, **f_kwargs: trace_calls(
                    *f_args, 
                    file=file, 
                    target_filename=target_filename, 
                    max_bytes=max_bytes, 
                    stop_flag=stop_tracing_flag, 
                    **f_kwargs
                ))
                # Call the original function
                result = func(*args, **kwargs)
                return result
            finally:
                sys.settrace(original_trace_function)
        return wrapper
    return decorator