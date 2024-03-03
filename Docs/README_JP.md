# BitNet

[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

論文「BitNet: Scaling 1-bit Transformers for Large Language Models」からの線形手法とモデルのPyTorch実装です。

[論文リンク:](https://arxiv.org/pdf/2310.11453.pdf) 

BitLinear = テンソル -> レイヤーノーム -> 二値化 -> 絶対最大量子化 -> 逆量子化

「BitNetアーキテクチャの実装は非常にシンプルで、Transformer内の線形射影（つまり、PyTorchのnn.Linear）を置換するだけです。」 -- BitNetは実装が本当に簡単で、線形モジュールをBitLinearモジュールに交換するだけです！
## **ニュース**  
- BitNet Transformerは、Wikipediaの小さな1GBデータセットであるenwiki8でトレーニングする`train.py`ファイルを使用してトレーニングされました：[こちらがリンクです](https://drive.google.com/file/d/1gBuZRFBqMV3cVD902LXA_hmZl4e0dLyY/view) 
- **新しい反復**  🔥 論文「[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) 」からの全く新しい反復があります。現在実装中です。Agoraのdiscordに参加して貢献しましょう！ [こちらで参加](https://discord.gg/hFzevCjG8c) 
- **新しい最適化**  最初の`BitLinear`が最適化され、注意メカニズムにBitLinearを実装するBit Attention `BitMGQA`を新たに得ました。Multi Grouped Query Attentionは、その高速なデコーディングと長いコンテキスト処理により、最高の注意と広く認識されています。Frankによる使いやすい実装に感謝します！
## 謝辞
- Dimitry, Nullonixによる分析、コードレビュー、および改訂
- トレーニング用に4080を提供してくれたVyom！
## インストール

`pip install bitnet`
## 使用方法:
### `BitLinear`
- 論文の主な革新であるBitLinearレイヤーの例：

```python
import torch

from bitnet import BitLinear

# 入力
x = torch.randn(10, 512)

# BitLinearレイヤー
layer = BitLinear(512, 400)

# 出力
y = layer(x)

print(y)
```

---
### `BitNetTransformer`
- MHAとBitFeedforwardsを備えた図に記載されている通りに完全に実装されたTransformer
- テキストだけでなく、画像やビデオ、オーディオ処理にも利用可能
- 勾配の流れのための残差とスキップ接続を完備

```python
# 必要なライブラリをインポート
import torch
from bitnet import BitNetTransformer

# 整数のランダムテンソルを作成
x = torch.randint(0, 20000, (1, 1024))

# BitNetTransformerモデルを初期化
bitnet = BitNetTransformer(
    num_tokens=20000,  # 入力のユニークなトークン数
    dim=1024,  # 入力および出力エンベディングの次元
    depth=6,  # トランスフォーマーレイヤーの数
    heads=8,  # 注意ヘッドの数
    ff_mult=4,  # フィードフォワードネットワーク内の隠れ層の次元の倍数
)

# テンソルをトランスフォーマーモデルを通して渡す
logits = bitnet(x)

# 出力の形状を印刷
print(logits)
```


### `BitAttention`

このAttentionは、デフォルトの線形射影の代わりにBitLinearを使用するように修正されました。また、通常のマルチヘッドアテンションの代わりにMulti-Grouped Query Attentionを使用しています。これにより、より高速なデコーディングとより長いコンテキスト処理が可能になります。

```python
import torch
from bitnet import BitMGQA

# 形状が(1, 10, 512)のランダムテンソルを作成
x = torch.randn(1, 10, 512)

# 入力サイズ512、注意ヘッド8、レイヤー4のBitMGQAモデルのインスタンスを作成
gqa = BitMGQA(512, 8, 4)

# 入力テンソルをBitMGQAモデルを通して渡し、出力と注意重みを取得
out, _ = gqa(x, x, x, need_weights=True)

# 出力テンソルと注意テンソルの形状を印刷
print(out)
```


### `BitFeedForward`
- BitLinearとGELUを使用した図に示されているフィードフォワード：
- 線形 -> GELU -> 線形
- より良いffnのためにドロップアウトやレイヤーノーム、その他のレイヤーを追加できます

```python
import torch
from bitnet
```
