# DecoupledLionW Optimizer

## Overview and Introduction

`DecoupledLionW` is a PyTorch optimizer designed to improve training performance and convergence for deep learning models. It is an extension of the Lion optimizer, which incorporates decoupled weight decay and a momentum-based update rule. 

The optimizer utilizes the Adam-like update rule, where the weight decay is applied separately from the gradient update. This is crucial as it helps prevent overfitting, improves generalization, and aids faster convergence and smoother optimization.

### Key Concepts:

- **Weight Decay:** Reduces the magnitude of the model's weights, preventing overfitting and improving generalization.
- **Momentum Update:** An interpolation between the current gradient and the previous momentum state, allowing for faster convergence and smoother optimization.
- **Momentum Decay:** Gradually reduces the momentum term over time, preventing it from becoming too large and destabilizing the optimization process.

## Class Definition

```python
class DecoupledLionW(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            weight_decay: float = 0.0,
    ):
```

### Parameters

- `params` (iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
- `lr` (float, optional): Learning rate. Default: 1e-4.
- `betas` (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Default: (0.9, 0.99).
- `weight_decay` (float, optional): Weight decay (L2 penalty). Default: 0.

### Attributes

- `metric_functions`: A dictionary of lambda functions to compute various metrics like L2 norm of moments, parameters, updates, and gradients, as well as cosine similarity between updates and gradients.

## Functionality and Usage

### `lionw` Method

This static method is responsible for applying the weight decay, momentum update, and momentum decay.

```python
@staticmethod
def lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2) -> None:
```

#### Parameters

- `p` (Tensor): Parameter tensor.
- `grad` (Tensor): Gradient tensor.
- `exp_avg` (Tensor): Exponential moving average of gradient values.
- `lr` (float): Learning rate.
- `initial_lr` (float): Initial learning rate.
- `wd` (float): Weight decay.
- `beta1` (float): Exponential decay rate for the first moment estimates.
- `beta2` (float): Exponential decay rate for the second moment estimates.

### `step` Method

Performs a single optimization step.

```python
@torch.no_grad()
def step(self, closure: Optional[Callable] = None):
```

#### Parameters

- `closure` (callable, optional): A closure that reevaluates the model and returns the loss.

#### Returns

- `loss` (float, optional): The loss value if `closure` is provided. None otherwise.

### `pre_reduce_metrics` Method

This method preprocesses the metrics before reduction across nodes.

```python
def pre_reduce_metrics(self, optimizer_metrics):
```

#### Parameters

- `optimizer_metrics` (dict): A dictionary containing the optimizer metrics.

#### Returns

- `optimizer_metrics` (dict): The pre-processed optimizer metrics.

### `report_per_parameter_metrics` Method

This method reports the per-parameter metrics.

```python
def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
```

#### Parameters

- `param` (Tensor): Parameter tensor.
- `name` (str): Name of the parameter.
- `optimizer_metrics` (dict): A dictionary containing the optimizer metrics.

#### Returns

- `optimizer_metrics` (dict): The optimizer metrics with the reported per-parameter metrics.

## Usage Examples

```python
from zeta import x
import torch

# Define model parameters
params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Define optimizer
optimizer = DecoupledLionW(params, lr=0.1, betas=(0.9, 0.999), weight_decay=0.01)

# Define loss function
loss_fn = torch.nn.MSELoss()

# Forward pass
output = x(params)
target = torch.tensor([0.0, 1.0, 2.0])
loss = loss_fn(output, target)

# Backward pass
loss.backward()

# Optimization step
optimizer.step()
```

## Mathematical Formula

The update rule of the optimizer can be represented by the following formula:

\[ p = p - \alpha \cdot \text{sign}(\beta_1 \cdot m + (1-\beta_1) \cdot g) - \eta \cdot wd \]

Where:

- \( p \) is the parameter.
- \( \alpha \) is the learning rate.
- \( \beta_1 \) is the exponential decay rate for the first moment estimates.
- \( m \) is the momentum (exponential moving average of gradient values).
- \( g \) is the gradient.
- \( \eta \) is the decay factor.
- \( wd \) is the weight decay.

## Additional Information and Tips

- A high value of `weight_decay` can lead to a large reduction in the model's weights on every step. Ensure to use an appropriate value for your specific use case.
- The optimizer supports both single-node and multi-node distributed training, enabling efficient training on parallel computing environments.
