# Documentation for `Trainer` Module from Zeta Library

---

## Introduction

The `Trainer` module from the Zeta library provides an easy-to-use, flexible, and scalable approach to training deep learning models. By abstracting away many of the lower-level details of training, including distributed training, gradient accumulation, and model checkpointing, `Trainer` allows developers to focus on the high-level aspects of model development and experimentation.

This module also integrates seamlessly with the HuggingFace `Accelerator` to enable mixed precision training, GPU acceleration, and distributed training across multiple nodes or GPUs.

---

## `Trainer` Class Definition

```python
def Trainer(
        gradient_accumulate_every: int = None, 
        batch_size: int = None, 
        seq_len: int = None,
        entity_name: str = None,
        model = None,
        use_fsdp: bool = False,
        use_activation_checkpointing: bool = False,
        learning_rate = None,
        seed = None,
        use_pretokenized: bool = False,
        resume_from_checkpoint = None,
        checkpointing_steps = None,
        output_dir = None,
        weight_decay = None,
        use_deepspeed = None
    ):
```

### Parameters

- `gradient_accumulate_every` (`int`, optional): Specifies how often to accumulate gradients. Default: `None`.
- `batch_size` (`int`, optional): Specifies the batch size for training. Default: `None`.
- `seq_len` (`int`, optional): Sequence length for model inputs. Default: `None`.
- `entity_name` (`str`, optional): Name of the entity for logging purposes. Default: `None`.
- `model`: The model to train. No default value.
- `use_fsdp` (`bool`, optional): Whether or not to use Fully Sharded Data Parallelism (FSDP). Default: `False`.
- `use_activation_checkpointing` (`bool`, optional): Use activation checkpointing to save memory during training. Default: `False`.
- `learning_rate`: The learning rate for training. No default value.
- `seed`: Random seed for reproducibility. No default value.
- `use_pretokenized` (`bool`, optional): Whether to use pre-tokenized data. Default: `False`.
- `resume_from_checkpoint`: Path to a checkpoint to resume training from. Default: `None`.
- `checkpointing_steps`: How often to save model checkpoints. Default: `None`.
- `output_dir`: Directory to save final trained model and checkpoints. Default: `None`.
- `weight_decay`: Weight decay value for regularization. No default value.
- `use_deepspeed`: Whether to use deepspeed for training optimization. Default: `None`.

---

## Functionality and Usage

The primary function of the `Trainer` module is to handle the training process, including data loading, optimization, and model updates. It leverages HuggingFace's `Accelerator` to provide accelerated training on GPUs and distributed environments.

Here are the primary steps:

1. Initialization of the `Accelerator` for GPU training and gradient accumulation.
2. Model and optimizer initialization.
3. Loading datasets and setting up data loaders.
4. Training loop with gradient accumulation and model checkpointing.
5. Save the final trained model.

### Code Examples

**1. Basic Usage**

```python
from zeta import Trainer

model = ... # Your model definition here
Trainer(
    gradient_accumulate_every=2,
    batch_size=32,
    seq_len=128,
    model=model,
    learning_rate=0.001,
    seed=42,
    output_dir='./models/'
)
```

**2. Resuming Training from a Checkpoint**

```python
from zeta import Trainer

model = ... # Your model definition here
Trainer(
    gradient_accumulate_every=2,
    batch_size=32,
    seq_len=128,
    model=model,
    learning_rate=0.001,
    seed=42,
    resume_from_checkpoint='./models/checkpoint.pt',
    output_dir='./models/'
)
```

**3. Using FSDP and Activation Checkpointing**

```python
from zeta import Trainer

model = ... # Your model definition here
Trainer(
    gradient_accumulate_every=2,
    batch_size=32,
    seq_len=128,
    model=model,
    use_fsdp=True,
    use_activation_checkpointing=True,
    learning_rate=0.001,
    seed=42,
    output_dir='./models/'
)
```

---

## Mathematical Description

Given a dataset \( D \) consisting of data points \( \{ (x_1, y_1), (x_2, y_2), ... (x_N, y_N) \} \), the trainer aims to minimize the loss function \( L \) with respect to model parameters \( \theta \):

\[ \theta^* = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} L(f(x_i; \theta), y_i) \]



where \( f \) is the model's prediction function.

---

## Conclusions

The `Trainer` module from Zeta library streamlines the training process by abstracting away many complexities, making it a valuable tool for developers at all experience levels. Whether you are training a simple model or a complex architecture in a distributed environment, the `Trainer` module offers the flexibility and ease-of-use to get your models trained efficiently.