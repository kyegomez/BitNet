[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# BitNet
![bitnet](/bitnet.png)
PyTorch Implementation of the linear methods and model from the paper "BitNet: Scaling 1-bit Transformers for Large Language Models"

[Paper link:](https://arxiv.org/pdf/2310.11453.pdf)

BitLinear = tensor -> layernorm -> Binarize -> abs max quantization -> dequant

"The implementation of the BitNet architecture is quite simple, requiring only the replacement of linear projections (i.e., nn.Linear in PyTorch) in the Transformer. " -- BitNet is really easy to implement just swap out the linears with the BitLinear modules! 

## **NEWS**
- **New Iteration** 🔥 There is an all-new iteration from the paper "[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)", we're implementing it now. Join the Agora discord and contribute! [Join Here](https://discord.gg/hFzevCjG8c)
- **New Optimizations** The first `BitLinear` has been optimized and we now have a Bit Attention `BitMGQA` That implements BitLinear into the attention mechanism. Multi Grouped Query Attention is also widely recognized as the best attention for its fast decoding and long context handling, thanks to Frank for his easy to use implementation!
- **BitLinear 1.5 Launch 🔥**: The new BitLinear 1.5 is still in progress 🔥 [Here is the file]() There are still some bugs like with the dequantization algorithm and we still need to replace the multiplication with elementwisw addition, if you could help with this, this would be amazing.
- **NOTICE**: A model obviously needs to be finetuned from scratch to use BitLinear, just changing the linear methods in an already trained model isn't going to work. Finetune or train from scratch.

## Appreciation
- Dimitry, Nullonix for analysis and code review and revision
- Vyom, for providing 4080 to train!

## Installation
```bash
pip3 install bitnet
```

## Usage
We have a vast selection of example scripts here and in the [examples folder:](/examples/), let me know if you want assistance with a use-case in the discord!


### `BitLinear`
- Example of the BitLinear layer which is the main innovation of the paper!
```python
import torch

from bitnet import BitLinear

# Input
x = torch.randn(10, 1000, 512)

# BitLinear layer
layer = BitLinear(512, 400)

# Output
y = layer(x)

print(y)

```

### BitLinearNew
```python
import torch
from bitnet import BitLinearNew

# Create a random tensor of shape (16, 10)
x = torch.randn(16, 1000, 512)

# Create an instance of the BitLinearNew class with input size 10, output size 20, and 2 groups
layer = BitLinearNew(
    512,
    20,
)

# Perform a forward pass through the BitLinearNew layer with input x
output = layer(x)

# Print the output tensor
print(output)
print(output.shape)
```
----

### `BitNetTransformer`
- Fully implemented Transformer as described in the diagram with MHA, and BitFeedforwards
- Can be utilized not just for text but for images and maybe even video or audio processing
- Complete with residuals and skip connections for gradient flow

```python
# Import the necessary libraries
import torch
from bitnet import BitNetTransformer

# Create a random tensor of integers
x = torch.randint(0, 20000, (1, 1024))

# Initialize the BitNetTransformer model
bitnet = BitNetTransformer(
    num_tokens=20000,  # Number of unique tokens in the input
    dim=1024,  # Dimension of the input and output embeddings
    depth=6,  # Number of transformer layers
    heads=8,  # Number of attention heads
    ff_mult=4,  # Multiplier for the hidden dimension in the feed-forward network
)

# Pass the tensor through the transformer model
logits = bitnet(x)

# Print the shape of the output
print(logits)

```


### `BitAttention`
This Attention has been modified to use BitLinear instead of the default linear projection. It's also using Multi-Grouped Query Attention instead of regular multi-head attention for faster decoding and longer context handling.

```python
import torch
from bitnet import BitMGQA

# Create a random tensor of shape (1, 10, 512)
x = torch.randn(1, 10, 512)

# Create an instance of the BitMGQA model with input size 512, 8 attention heads, and 4 layers
gqa = BitMGQA(512, 8, 4)

# Pass the input tensor through the BitMGQA model and get the output and attention weights
out, _ = gqa(x, x, x, need_weights=True)

# Print the shapes of the output tensor and attention tensor
print(out)
```

### `BitFeedForward`
- Feedforward as shown in the diagram with BitLinear and a GELU:
- Linear -> GELU -> Linear
- You can add dropouts, or layernorms, or other layers for a better ffn

```python
import torch
from bitnet import BitFeedForward

# Create a random input tensor of shape (10, 512)
x = torch.randn(10, 512)

# Create an instance of the BitFeedForward class with the following parameters:
# - input_dim: 512
# - hidden_dim: 512
# - num_layers: 4
# - swish: True (use Swish activation function)
# - post_act_ln: True (apply Layer Normalization after each activation)
# - dropout: 0.1 (apply dropout with a probability of 0.1)
ff = BitFeedForward(512, 512, 4, swish=True, post_act_ln=True, dropout=0.1)

# Apply the BitFeedForward network to the input tensor x
y = ff(x)

# Print the shape of the output tensor y
print(y)  # torch.Size([10, 512])
```

## Inference
```python
from bitnet import BitNetInference

bitnet = BitNetInference()
bitnet.load_model("../model_checkpoint.pth")  # Download model
output_str = bitnet.generate("The dog jumped over the ", 512)
print(output_str)
```

## Huggingface Usage
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bitnet import replace_linears_in_hf

# Load a model from Hugging Face's Transformers
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Replace Linear layers with BitLinear
replace_linears_in_hf(model)

# Example text to classify
text = "Replace this with your text"
inputs = tokenizer(
    text, return_tensors="pt", padding=True, truncation=True, max_length=512
)

# Perform inference
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

# Process predictions
predicted_class_id = predictions.argmax().item()
print(f"Predicted class ID: {predicted_class_id}")

# Optionally, map the predicted class ID to a label, if you know the classification labels
# labels = ["Label 1", "Label 2", ...]  # Define your labels corresponding to the model's classes
# print(f"Predicted label: {labels[predicted_class_id]}")
```


## Drop in Replacement for Pytorch Models
```python
import torch
from torch import nn
from bitnet import replace_linears_in_pytorch_model

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
)

print("Before replacement:")
print(model)

# Replace nn.Linear with BitLinear
replace_linears_in_pytorch_model(model)

print("After replacement:")
print(model)

# Now you can use the model for training or inference
# For example, pass a random input through the model
input = torch.randn(1, 10)
output = model(input)
```


### Optimized Cuda Kernel
`python setup.py build_ext --inplace`

```python
import torch
import gemm_lowbit_ext  # This imports the compiled module

# Example usage
a = torch.randn(10, 20, dtype=torch.half, device='cuda')  # Example tensor
b = torch.randn(20, 30, dtype=torch.half, device='cuda')  # Example tensor
c = torch.empty(10, 30, dtype=torch.half, device='cuda')  # Output tensor

w_scale = 1.0  # Example scale factor
x_scale = 1.0  # Example scale factor

# Call the custom CUDA GEMM operation
gemm_lowbit_ext.gemm_lowbit(a, b, c, w_scale, x_scale)

print(c)  # View the result

```


## `BitLora`
Implementation of BitLora!

```python
import torch
from bitnet import BitLora

# Random text tensor
x = torch.randn(1, 12, 200)

# Create an instance of the BitLora model
model = BitLora(in_features=200, out_features=200, rank=4, lora_alpha=1)

# Perform the forward pass
out = model(x)

# Print the shape of the output tensor
print(out.shape)
```


## BitMamba
```python
import torch
from bitnet import BitMamba

# Create a tensor of size (2, 10) with random values between 0 and 100
x = torch.randint(0, 100, (2, 10))

# Create an instance of the BitMamba model with input size 512, hidden size 100, output size 10, and depth size 6
model = BitMamba(512, 100, 10, 6, return_tokens=True)

# Pass the input tensor through the model and get the output
output = model(x)

# Print the output tensor
print(output)

# Print the shape of the output tensor
print(output.shape)

```

## `BitMoE`

```python
import torch
from bitnet.bit_moe import BitMoE

# Create input tensor
x = torch.randn(2, 4, 8)

# Create BitMoE model with specified input and output dimensions
model = BitMoE(8, 4, 2)

# Forward pass through the model
output = model(x)

# Print the output
print(output)
```


### 1 Bit Vision Transformers
This idea came to me out of nowhere but it seems to be pretty fun, as you can leverage bitlinear for vision tasks for ultra-compression. It would be nice to train this on imagenet if you could make a script, we'll provide the compute. Then the next stage would be to train a join vision language model gpt-4o

```python
import torch
from bitnet import OneBitViT

# Create an instance of the OneBitViT model
v = OneBitViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
)

# Generate a random image tensor
img = torch.randn(1, 3, 256, 256)

# Pass the image through the OneBitViT model to get predictions
preds = v(img)  # (1, 1000)

# Print the predictions
print(preds)

```

# License
MIT

# Citation
```bibtex
@misc{2310.11453,
Author = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
Title = {BitNet: Scaling 1-bit Transformers for Large Language Models},
Year = {2023},
Eprint = {arXiv:2310.11453},
}

```


# Todo
- [x] Double check BitLinear implementation and make sure it works exactly as in paper 
- [x] Implement training script for `BitNetTransformer`
- [x] Train on Enwiki8, copy and past code and data from Lucidrains repos
- [x] Benchmark performance
- [x] Look into Straight Through Estimator for non-differentiable backprop
- [x] Implement BitFeedForward
- [x] Clean up codebase 
- [x] Add unit tests for each module
- [x] Implement the new BitNet1.5b from the [paper](https://arxiv.org/abs/2402.17764)
- [ ] Implement the BitNet15b in Cuda
- [ ] Implement the low bit gemm cuda kernel 