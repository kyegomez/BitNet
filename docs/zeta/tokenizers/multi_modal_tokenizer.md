# **Documentation for Zeta Library's MultiModalTokenizer Class**

---

## **Introduction and Overview**

The `MultiModalTokenizer` class is part of the Zeta Library, designed to provide tokenization capabilities for both text and image data. This enables more seamless integration and utilization of multimodal (text and image) data, especially when used with models that can handle such information simultaneously, like the CLIP model.

**Key Features**:

1. **Multimodal Tokenization**: Combines text and image tokenization within one unified class.
2. **Integration with Hugging Face Transformers**: Utilizes the `CLIPProcessor` for image tokenization and `AutoTokenizer` for text tokenization.
3. **Special Tokens for Image Segmentation**: Uses special tokens `<image>` and `</image>` to denote image token boundaries within text.
4. **Error Handling**: Implements comprehensive error handling and logging to ensure robustness.

---

## **Class Definition**

### **MultiModalTokenizer**

```python
class MultiModalTokenizer:
    """
    A tokenizer class for the kosmos model

    Attributes:
        processor(CLIPProcessor): The processor to tokenize images.
        tokenizer(AutoTokenizer): The tokenizer to tokenize text.
        im_idx(int): The Index of the "<image>" token.
        im_end_idx(int): The index of the "</image>" token.
    """
```

#### **Parameters**:

- **max_length (int, optional)**: Maximum length of the tokenized sequence. Defaults to 8192.

#### **Attributes**:

- **processor (CLIPProcessor)**: The processor used to tokenize images.
- **tokenizer (AutoTokenizer)**: The tokenizer used to tokenize text.
- **im_idx (int)**: Index of the `<image>` token.
- **im_end_idx (int)**: Index of the `</image>` token.

---

## **Methods**

### **1. tokenize_texts**

```python
def tokenize_texts(self, texts: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize given texts.

    Args:
        texts (str): The text to be tokenized.

    Returns:
        A tuple containing the tokenized texts and only the text tokens.
    """
```

### **2. tokenize_images**

```python
def tokenize_images(self, images) -> torch.Tensor:
    """
    Tokenizes given images.

    Args:
        images: The images to be tokenized.

    Returns:
        The tokenized images.
    """
```

### **3. tokenize**

```python
def tokenize(self, sample) -> Dict[str, torch.Tensor]:
    """
    Tokenizes given sample.

    Args:
        sample: The sample to be tokenized.

    Returns:
        A dictionary containing the tokenized text tokens, images, labels, and attention mask.
    """
```

---

## **Usage Examples**

### **Example 1: Tokenizing Texts**

```python
from zeta import MultiModalTokenizer
import torch

tokenizer = MultiModalTokenizer()
texts = ["Hello World", "Zeta Library is great!"]
tokenized_texts, only_texts = tokenizer.tokenize_texts(texts)
print(tokenized_texts)
print(only_texts)
```

### **Example 2: Tokenizing Images**

```python
from zeta import MultiModalTokenizer
import torch

tokenizer = MultiModalTokenizer()
images = torch.randn(2, 3, 224, 224)  # Assuming 2 random images of shape 3x224x224
tokenized_images = tokenizer.tokenize_images(images)
print(tokenized_images)
```

### **Example 3: Tokenizing Multimodal Data**

```python
from zeta import MultiModalTokenizer
import torch

tokenizer = MultiModalTokenizer()
sample = {
    "target_text": ["Hello World", "Zeta Library is great!"],
    "image": torch.randn(2, 3, 224, 224)
}
tokenized_data = tokenizer.tokenize(sample)
print(tokenized_data)
```

---

## **Mathematical Overview**

Given a text sequence \( T \) of length \( n \) and an image \( I \) represented by a tensor of shape \( C \times H \times W \), where \( C \) is the number of channels, \( H \) is the height, and \( W \) is the width:

1. The tokenized text, \( T' \), is represented as:
   \[ T' = [<s>, <image>, </image>, T_{1}, T_{2}, ..., T_{n}, </s>] \]

2. The tokenized image, \( I' \), is processed using the CLIP processor to obtain a tensor representation.

3. When both text and image data are tokenized using the `tokenize` method, the output contains both \( T' \) and \( I' \) with their respective attention masks.

---

## **Additional Tips**

- Ensure you have the required model weights and configurations for the specified pretrained models ("laion/CLIP-ViT-L-14-laion2B-s32B-b82K" and "EleutherAI/gpt-neox-20b") downloaded or accessible from the Hugging Face Model Hub.
  
- Handle potential tokenization errors gracefully using try-except blocks, as demonstrated in the provided methods.

---

## **References and Resources**

1. CLIP: Connecting Vision and Language with Reinforced Loss - OpenAI: [Link](https://openai.com/blog/clip/)
2. Hugging Face's Transformers library: [Link](https://huggingface.co/transformers/)
3. Documentation on Special Tokens in Transformers: [Link](https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.add_special_tokens)

---