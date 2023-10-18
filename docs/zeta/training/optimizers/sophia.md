# SophiaG Optimizer for Zeta Library

## Overview

The SophiaG optimizer is designed to adaptively change learning rates during training, offering a combination of momentum-based acceleration and second-order Hessian-based adaptive learning rates. This optimizer is particularly useful for training deep neural networks and optimizing complex, non-convex loss functions. Key features include:

1. **Momentum**: Utilizes exponentially moving averages of gradients.
2. **Adaptive Learning Rate**: Adjusts the learning rate based on the second-order Hessian information.
3. **Regularization**: Applies weight decay to avoid overfitting.
4. **Optional Settings**: Allows for maximizing the loss function, customizable settings for capturable and dynamic parameters.

## Class Definition

```python
class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 capturable: bool = False, dynamic: bool = False):
```

### Parameters:

- `params` (iterable): Iterable of parameters to optimize.
- `lr` (float, default=1e-4): Learning rate.
- `betas` (Tuple[float, float], default=(0.965, 0.99)): Coefficients used for computing running averages of gradient and Hessian.
- `rho` (float, default=0.04): Damping factor for Hessian-based updates.
- `weight_decay` (float, default=1e-1): Weight decay factor.
- `maximize` (bool, default=False): Whether to maximize the loss function.
- `capturable` (bool, default=False): Enable/Disable special capturing features.
- `dynamic` (bool, default=False): Enable/Disable dynamic adjustments of the optimizer.

## Usage and Functionality

### 1. Initialization

Upon initialization, the optimizer performs validation on its parameters and sets them as the default parameters for parameter groups.

```python
from zeta import SophiaG

optimizer = SophiaG(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-4)
```

### 2. Step Forward

The `.step()` method updates the model parameters. The function is decorated with `@torch.no_grad()` to avoid saving any more computation graphs for gradient computation.

```python
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

### 3. Update Hessian and Exponential Average

The optimizer has internal methods to update the Hessian and Exponential Moving Average (EMA) of the gradients, controlled by `betas`.

### 4. SophiaG Function

The core SophiaG function updates the parameters based on the gradient (`grad`), moving average (`exp_avg`), and Hessian (`hessian`). It uses the following update formula:

\[ \text{param} = \text{param} - \text{lr} \times \left( \text{beta}_1 \times \text{exp_avg} + \frac{(1-\text{beta}_1) \times \text{grad}}{( \text{beta}_2 \times \text{hessian} + (1-\text{beta}_2) )^{\rho}} \right) \]

## Usage Examples

### 1. Basic Usage:

```python
from zeta import SophiaG
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = SophiaG(model.parameters(), lr=0.01)
```

### 2. Customizing Betas and Learning Rate:

```python
from zeta import SophiaG
import torch

optimizer = SophiaG(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

### 3. Using with Weight Decay:

```python
from zeta import SophiaG

optimizer = SophiaG(model.parameters(), lr=0.01, weight_decay=1e-4)
```

## Additional Information and Tips

- Make sure that the parameters passed are compatible with the model you are using.
- To maximize the loss function (useful in adversarial training), set `maximize=True`.

## Common Issues

- If sparse gradients are involved, the SophiaG optimizer is not applicable.

## References and Resources

- [Adaptive Learning Rates](https://arxiv.org/pdf/1609.04747)
- [Zeta Documentation](https://zeta.apac.ai)

For further questions or issues, visit our [GitHub repository](https://github.com/kyegomez/zeta).
