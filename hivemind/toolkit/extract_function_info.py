
import inspect
import dataclasses

@dataclasses.dataclass
class FunctionInfo:
    name: str
    signature: str
    docstring: str
    source_code: str
    param_docstrings: dict
    return_docstring: str

def extract_function_info(func):
    name = func.__name__
    signature = str(inspect.signature(func))
    docstring = inspect.getdoc(func)
    source_code = inspect.getsource(func)

    sig = inspect.signature(func)
    param_docstrings = {name: inspect.getdoc(param.annotation) for name, param in sig.parameters.items() if param.annotation is not inspect.Parameter.empty}
    return_docstring = inspect.getdoc(sig.return_annotation) if sig.return_annotation is not inspect.Signature.empty else None

    func_info = FunctionInfo(name, signature, docstring, source_code, param_docstrings, return_docstring)
    
    return func_info


# Test the function with a function
def add_numbers(a: int, b: int) -> int:
    '''
    This function takes two integers as input and returns their sum.

    Parameters:
    a (int): The first number
    b (int): The second number

    Returns:
    int: The sum of the two numbers
    '''
    result = a + b
    print(f"The sum of {a} and {b} is {result}")
    return result

test_func_info = extract_function_info(add_numbers)
print(test_func_info)
