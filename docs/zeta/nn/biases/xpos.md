# XPOS Module Documentation
-------------------------

### Architecture

The XPOS module is a part of a neural network model and is implemented as a subclass of `torch.nn.Module`. It consists of several functions and a class that work together to apply rotary positional embeddings to an input tensor.

### Purpose

The purpose of the XPOS module is to incorporate positional information into the input tensor of a neural network model. It achieves this by generating fixed positional embeddings and applying them to the input tensor using rotary positional encoding techniques. This allows the model to capture the sequential order and relative positions of the input elements, which can be beneficial for tasks such as natural language processing and time series analysis.

### Functions and Methods

1.  `fixed_pos_embedding(x)`: Generates fixed positional embeddings for the input tensor.

    -   Args:
        -   `x` (torch.Tensor): Input tensor of shape `(seq_len, dim)`.
    -   Returns:
        -   `sin` (torch.Tensor): Sine positional embeddings of shape `(seq_len, dim)`.
        -   `cos` (torch.Tensor): Cosine positional embeddings of shape `(seq_len, dim)`.
2.  `rotate_every_two(x)`: Rearranges the elements of the input tensor by rotating every two elements.

    -   Args:
        -   `x` (torch.Tensor): Input tensor of shape `(batch_size, seq_len, dim)`.
    -   Returns:
        -   `x` (torch.Tensor): Rearranged tensor of shape `(batch_size, seq_len, dim)`.
3.  `duplicate_interleave(m)`: Duplicates a matrix while interleaving the copy.

    -   Args:
        -   `m` (torch.Tensor): Input matrix.
    -   Returns:
        -   `m` (torch.Tensor): Duplicated and interleaved matrix.
4.  `apply_rotary_pos_emb(x, sin, cos, scale=1)`: Applies rotary positional embeddings to the input tensor.

    -   Args:
        -   `x` (torch.Tensor): Input tensor of shape `(batch_size, seq_len, dim)`.
        -   `sin` (torch.Tensor): Sine positional embeddings of shape `(seq_len, dim)`.
        -   `cos` (torch.Tensor): Cosine positional embeddings of shape `(seq_len, dim)`.
        -   `scale` (float): Scaling factor for the positional embeddings (default: 1).
    -   Returns:
        -   `x` (torch.Tensor): Tensor with applied rotary positional embeddings.
5.  `XPOS(head_dim, scale_base=512)`: XPOS module class.

    -   Args:
        -   `head_dim` (int): Dimensionality of the input tensor.
        -   `scale_base` (int): Base value for scaling the positional embeddings (default: 512).
    -   Methods:
        -   `forward(x, offset=0, downscale=False)`: Forward pass of the XPOS module.
            -   Args:
                -   `x` (torch.Tensor): Input tensor of shape `(batch_size, seq_len, dim)`.
                -   `offset` (int): Offset value for positional embeddings (default: 0).
                -   `downscale` (bool): Boolean indicating whether to downscale the positional embeddings (default: False).
            -   Returns:
                -   `x` (torch.Tensor): Tensor with applied rotary positional embeddings.

### Usage Examples

1.  Applying XPOS module to an input tensor:

    ```
    import torch
    from xpos import XPOS

    # Create an instance of the XPOS module
    xpos = XPOS(head_dim=256)

    # Generate a random input tensor
    x = torch.randn(1, 10, 256)

    # Apply the XPOS module to the input tensor
    output = xpos(x)
    ```


2.  Applying XPOS module with offset and downscaling:

    ```
    import torch
    from zeta import XPOS

    # Create an instance of the XPOS module
    xpos = XPOS(head_dim=512)

    # Generate a random input tensor
    x = torch.randn(1, 20, 512)

    # Apply the XPOS module to the input tensor with offset and downscaling
    output = xpos(x, offset=2, downscale=True)
    ```
3.  Using the individual functions of the XPOS module:

    ```
    import torch
    from zeta import fixed_pos_embedding, apply_rotary_pos_emb

    # Generate fixed positional embeddings
    scale = torch.randn(10, 256)
    sin, cos = fixed_pos_embedding(scale)

    # Apply rotary positional embeddings to an input tensor
    x = torch.randn(1, 10, 256)
    output = apply_rotary_pos_emb(x, sin, cos, scale=0.5)
    ```

Note: The above examples assume that the `xpos.py` file