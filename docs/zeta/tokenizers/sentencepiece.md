# SentencePieceTokenizer

`SentencePieceTokenizer` is a class for tokenizing and detokenizing text using a pre-trained SentencePiece model. The SentencePiece model is a unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation tasks where the vocabulary size is predetermined prior to the neural model training. This class is a part of the zeta library which is a collection of various utility functions and classes for Natural Language Processing tasks.

## Introduction

Tokenization is a crucial step in many natural language processing tasks. It involves splitting a piece of text into smaller units, called tokens. These tokens can be as small as characters or as large as words. The `SentencePieceTokenizer` class provides an efficient and easy-to-use way to tokenize and detokenize text using a SentencePiece model.

The SentencePiece model is trained to find the best tokenization by dynamically adjusting the size and boundary of tokens. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) and unigram language model with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.

## Class Definition

```python
class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        ...
```

### Parameters:

- `model_path (str)`: The path to the pre-trained SentencePiece model. It should be a file with `.model` extension.

### Attributes:

- `n_words (int)`: The vocabulary size of the SentencePiece model.
- `bos_id (int)`: The token ID for the beginning of sentence token.
- `eos_id (int)`: The token ID for the end of sentence token.
- `pad_id (int)`: The token ID for the padding token.
- `prefix_id (int, optional)`: The token ID for the prefix token.
- `middle_id (int, optional)`: The token ID for the middle token.
- `suffix_id (int, optional)`: The token ID for the suffix token.
- `eot_id (int, optional)`: The token ID for the end of text token.

## Methods

### `encode`

```python
def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    ...
```

Encodes a string into a list of integer token IDs.

#### Parameters:

- `s (str)`: The string to be encoded.
- `bos (bool)`: Whether to add the beginning of sentence token at the start.
- `eos (bool)`: Whether to add the end of sentence token at the end.

#### Returns:

- `List[int]`: A list of integer token IDs.

### `decode`

```python
def decode(self, t: List[int]) -> str:
    ...
```

Decodes a list of integer token IDs into a string.

#### Parameters:

- `t (List[int])`: A list of integer token IDs to be decoded.

#### Returns:

- `str`: The decoded string.

### `encode_infilling`

```python
def encode_infilling(self, s: str) -> List[int]:
    ...
```

Encodes a string without an implicit leading space.

#### Parameters:

- `s (str)`: The string to be encoded.

#### Returns:

- `List[int]`: A list of integer token IDs.

### `decode_infilling`

```python
def decode_infilling(self, t: List[int]) -> str:
    ...
```

Decodes a list of integer token IDs into a string without an implicit leading space.

#### Parameters:

- `t (List[int])`: A list of integer token IDs to be decoded.

#### Returns:

- `str`: The decoded string.

## Usage Examples

### Example 1:

```python
from zeta import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer(model_path='path/to/your/model.model')
text = "Hello, world!"
tokens = tokenizer.encode(text, bos=True, eos=True)
print(tokens)
# [2, 284, 16, 250, 13, 849, 4, 3]

decoded_text = tokenizer.decode(tokens)
print(decoded_text)
# "Hello, world!"
```

### Example 2:

```python
from zeta import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer(model_path='path/to/your/model.model')
text = "Hello, world!"
tokens = tokenizer.encode_infilling(text)
print(tokens)
# [284, 16, 250, 13, 849, 4]

decoded_text = tokenizer.decode_infilling(tokens)
print(decoded_text)
# "Hello, world!"
```

### Example 3:

```python
from zeta import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer(model_path='path/to/your/model.model')
tokens = [2, 284, 16, 250, 13, 849, 4, 3]
decoded_text = tokenizer.decode(tokens)
print(decoded_text)
# "Hello, world!"
```

## Additional Information

- Make sure that the model file specified in `model_path` exists.
- The special tokens such as `<PRE>`, `<MID>`, `<SUF>`, `<EOT>` are optional and may not be present in all SentencePiece models.

## References and Resources

- [SentencePiece GitHub Repository](https://github.com/google/sentencepiece)
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Generation](https://arxiv.org/abs/1808.06226)

## Mathematical Formulation

The SentencePiece model uses the following mathematical formula for tokenization:

\[P(w) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})\]

Where:
- \(P(w)\) is the probability of the word \(w\).
- \(n\) is the number of subwords in the word \(w\).
- \(w_i\) is the \(i\)-th subword of \(w\).

The model is trained to maximize the likelihood of the training data, and the subwords are chosen to minimize the perplexity of the training data.