The Zeta framework provides developers with the ability to create State of The Art Models as simply and seamlessly as possible through **Modularity**, **Reliability**, **Use-Ability**, and **Speed**

Zeta not only helps developers harness the potential of LLMs and Multi-Modal Foundation Models but also enforces trust boundaries, schema validation, and tool activity-level permissions. By doing so, Zeta maximizes LLMs’ reasoning while adhering to strict policies regarding their capabilities.

Zeta’s design philosophy is based on the following tenets:

1. **Use-Ability**: Utilizing Zeta should feel like going for a swim in the ocean, seamless and fluid with pythonic methods and classes and error handling that signifies what steps to take next.
2. **Reliability**: Zeta puts every FLOP to work by harnessing ultra-reliable and high-performance designs for all functions and classes
3. **Speed**: Zeta is like the Lamborghini of ML Frames with simply unparalled speed.

## Quick Starts

### Using pip

Install **zeta**

```
pip3 install zeta 
```

## Unleash FlashAttention
With Zeta, you can unleash the best and highest performance attention mechanisms like `FlashAttention` and `MultiQueryAttention`, here's an example with Flash Attention

```python
import torch
from zeta import FlashAttention

q = torch.randn(2, 4, 6, 8)
k = torch.randn(2, 4, 10, 8)
v = torch.randn(2, 4, 10, 8)

attention = FlashAttention(causal=False, dropout=0.1, flash=False)
output = attention(q, k, v)

print(output.shape) 
```

## Unleash GPT-4 
On top of the SOTA Attention mechanisms we provide, we also provide rough implementation of some of the best neural nets ever made like `GPT4`, here's an example on how to utilize our implementation of GPT-4

```python
import torch
from zeta import GPT4, GPT4MultiModal

#text
text = torch.randint(0, 256, (1, 1024)).cuda()
img = torch.randn(1, 3, 256, 256)

gpt4_language = GPT4()

gpt4_language(x)

#multimodal GPT4

gpt4_multimodal = GPT4MultiModal()
gpt4_multimodal_output = gpt4_multimodal(text, img)

```

