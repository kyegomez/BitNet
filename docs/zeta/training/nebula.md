# Nebula

The `Nebula` class is a custom loss function class that dynamically determines the most suitable loss function for a given dataset based on certain characteristics of the dataset, such as sparsity, correlation, range of values, and user input. It is part of the `zeta` library and is built upon PyTorch's LossFunction class.

## Introduction

The purpose of the `Nebula` class is to help determine and cache the most suitable loss function for a given dataset without requiring the user to manually select one. This can be particularly useful in scenarios where the user is unsure of the most appropriate loss function to use or in automated systems where the type of problem (classification or regression) is not known a priori.

The `Nebula` class considers various characteristics of the data, such as whether the target values are integers, the sparsity of the target values, the correlation between predictions and target values, and any user or domain knowledge provided, to determine whether the problem is a classification or regression problem and subsequently select an appropriate loss function.

## Class Definition

```python
class Nebula(LossFunction):
    def __init__(self, domain_knowledge=None, user_input=None):
        ...
```

### Parameters

- `domain_knowledge` (str, optional): Domain knowledge about the problem. It can be either "classification" or "regression". Default is `None`.
- `user_input` (str, optional): User input about the problem type. It can be either "classification" or "regression". Default is `None`.

### Attributes

- `loss_function`: The determined loss function.
- `domain_knowledge`: Domain knowledge provided during initialization.
- `user_input`: User input provided during initialization.
- `loss_function_cache`: A cache for storing the determined loss function for a dataset.
- `unique_values_cache`: A cache for storing the unique values in the target variable `y_true`.
- `class_balance_cache`: A cache for storing the class balance in the target variable `y_true`.
- `logger`: A logger for logging information during the determination of the loss function.

## Functionality and Usage

The `Nebula` class is used to dynamically determine the most suitable loss function for a given dataset and cache the determined loss function for future use. The class analyzes the unique values, class balance, sparsity, and correlation of the target variable `y_true` and the predicted variable `y_pred` to determine whether the problem is a classification or regression problem and select an appropriate loss function.

### Method: `determine_loss_function`

```python
def determine_loss_function(self, y_pred, y_true):
    ...
```

This method determines the most suitable loss function based on the characteristics of `y_pred` and `y_true`.

#### Parameters

- `y_pred` (Tensor): The predicted values.
- `y_true` (Tensor): The ground truth values.

### Method: `__call__`

```python
def __call__(self, y_pred, y_true):
    ...
```

This method computes the loss using the determined loss function.

#### Parameters

- `y_pred` (Tensor): The predicted values.
- `y_true` (Tensor): The ground truth values.

#### Returns

- `Tensor`: The computed loss.

### Usage Examples

#### Example 1: Basic Usage

```python
from zeta import Nebula
import torch

# Initialize Nebula
nebula = Nebula()

# Generate some example data
y_pred = torch.randn(10, 5)
y_true = torch.randint(0, 5, (10,))

# Compute the loss
loss = nebula(y_pred, y_true)

print(loss)
```

#### Example 2: Providing Domain Knowledge

```python
from zeta import Nebula
import torch

# Initialize Nebula with domain knowledge
nebula = Nebula(domain_knowledge="classification")

# Generate some example data
y_pred = torch.randn(10, 5)
y_true = torch.randint(0, 5, (10,))

# Compute the loss
loss = nebula(y_pred, y_true)

print(loss)
```

#### Example 3: Providing User Input

```python
from zeta import Nebula
import torch

# Initialize Nebula with user input
nebula = Nebula(user_input="regression")

# Generate some example data
y_pred = torch.randn(10, 1)
y_true = torch.randn(10, 1)

# Compute the loss
loss = nebula(y_pred, y_true)

print(loss)
```

## Mathematical Formula

The `Nebula` class does not have a specific mathematical formula as it dynamically determines the most suitable loss function based on the characteristics of the data. However, the determined loss function will have its own mathematical formula, which can be found in the PyTorch documentation or the `zeta` library documentation.

## Additional Information and Tips

- The `Nebula` class caches the determined loss function, unique values, and class balance for a given dataset to avoid recomputing them in the future.
- If both `domain_knowledge` and `user_input` are provided, `domain_knowledge` will take precedence over `user_input`.
- The `Nebula` class uses the `logging` module to log information during the determination of the loss function. You can customize the logging settings by modifying the `logger` attribute.

