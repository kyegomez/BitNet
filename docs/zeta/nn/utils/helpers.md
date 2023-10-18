## Documentation

### Overview

The provided module comprises utility functions and classes to streamline specific operations with Python data structures and PyTorch models. The main aspects of the module are:

- Checking the existence of a value.
- Implementing custom call behavior through classes.
- Custom decorators for function calls.
- Dictionary manipulation.
- Initialization of PyTorch layer parameters.

### Functions and Classes

1. **exists(val: Any) -> bool**:  
   Checks if the provided value is not `None`.

2. **default(val: Any, d: Any) -> Any**:  
   Returns the value if it's not `None`; otherwise, it returns a default value.

3. **once(fn: Callable) -> Callable**:  
   A decorator ensuring that the function is only called once.

4. **eval_decorator(fn: Callable) -> Callable**:  
   A decorator for `torch.nn.Module` methods to switch the module to `eval` mode during the function call and revert to its original mode afterwards.

5. **cast_tuple(val: Any, depth: int) -> Tuple**:  
   Casts a value to a tuple with a specific depth.

6. **maybe(fn: Callable) -> Callable**:  
   A decorator that calls the function only if its first argument exists.

7. **always**:  
   A class that always returns the specified value when called.

8. **not_equals** and **equals**:  
   Classes that, when instantiated with a value, check if another value is (not) equal to the specified value.

9. **init_zero_(layer: nn.Module) -> None**:  
   Initializes the weights and biases of a torch layer to zero.

10. **pick_and_pop(keys: List[str], d: Dict) -> Dict**:  
   Extracts values from a dictionary based on provided keys.

11. **group_dict_by_key(cond: Callable, d: Dict) -> Tuple[Dict, Dict]**:  
   Groups dictionary keys based on a given condition.

12. **string_begins_with(prefix: str, str: str) -> bool**:  
   Checks if a string starts with a specific prefix.

13. **group_by_key_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]**:  
   Groups dictionary items by keys starting with a specific prefix.

14. **groupby_prefix_and_trim(prefix: str, d: Dict) -> Tuple[Dict, Dict]**:  
   Similar to `group_by_key_prefix` but also removes the prefix from keys.

### Usage Examples

1. **Using the `once` decorator**:

    ```python
    from zeta import once

    @once
    def greet():
        print("Hello, World!")

    greet()  # prints "Hello, World!"
    greet()  # Does nothing on the second call
    ```

2. **Using the `eval_decorator` with PyTorch**:

    ```python
    import torch.nn as nn
    from zeta import eval_decorator

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 10)

        @eval_decorator
        def predict(self, x):
            return self.layer(x)

    model = SimpleModel()
    input_tensor = torch.randn(1, 10)
    output = model.predict(input_tensor)  # Automatically switches to eval mode and back
    ```

3. **Dictionary Manipulation with Prefix Functions**:

    ```python
    from zeta import group_by_key_prefix
    
    sample_dict = {
        "user_name": "John",
        "user_age": 25,
        "order_id": 12345,
        "order_date": "2023-01-01"
    }

    user_data, order_data = group_by_key_prefix("user_", sample_dict)
    print(user_data)  # {'user_name': 'John', 'user_age': 25}
    print(order_data)  # {'order_id': 12345, 'order_date': '2023-01-01'}
    ```

This module is a collection of general-purpose utility functions and classes, making many common operations more concise. It's beneficial when working with PyTorch models and various data manipulation tasks.