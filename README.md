# Hiroyuki SLM

ひろゆき風の話し方をする Small Language Model (SLM) です。  
Qwen2.5-0.5B-Instruct をベースモデルとし、LoRA アダプターでひろゆき調の応答を生成するようにファインチューニングされています。  
4bit 量子化により、GPU メモリを抑えつつ動作可能です。

## プロジェクト概要

Hiroyuki SLM は、ひろゆきさん特有の冷静で論理的、かつ少しズレた視点からの返答を生成するチャットボットです。  
Unsloth を使用した効率的な学習と、PEFT/LoRA による軽量アダプター学習を採用しています。

## 特徴

- **ひろゆき風応答**: 冷静・論理的・皮肉めいた視点を再現
- **4-bit 量子化**: bitsandbytes によるメモリ効率化
- **LoRA アダプター**: 軽量なパラメーター効率的学習
- **FastAPI + Uvicorn**: 高速な REST API サーバー
- **非同期生成**: `asyncio` によるブロッキング回避
- **日本語特化**: 日本語での対話に最適化

## ファイル構成

```
/workspace/
├── main.py                 # エントリーポイント
├── api.py                  # FastAPI アプリケーション
├── slm_model.py            # HiroyukiSLM モデル実装
├── requirements.txt        # 依存パッケージ
├── hiroyuki_adapter/       # LoRA アダプター（学習済み重み）
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── data/                   # データセット用ディレクトリ
├── sh/                     # シェルスクリプト
│   ├── build.sh            # ビルドスクリプト
│   └── start.sh            # 起動スクリプト
└── README.md               # このファイル
```

## 必要環境

- Python 3.10+
- NVIDIA GPU（CUDA 対応）推奨
- VRAM: 4GB 以上（4bit 量子化時）

## インストール

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. 依存関係のインストール

```bash
# build.sh を使用してインストール
bash sh/build.sh
```

または手動でインストール：

```bash
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps -r requirements.txt
```

## 使い方

### API サーバーの起動

```bash
# start.sh を使用
bash sh/start.sh

# または直接実行
python main.py
```

サーバーはデフォルトで `http://0.0.0.0:8000` で起動します。

### API エンドポイント

#### 1. ヘルスチェック

```bash
GET /health
```

レスポンス例：
```json
{
  "status": "ok"
}
```

#### 2. チャット

```bash
POST /chat
Content-Type: application/json

{
  "message": "あなたのメッセージ"
}
```

リクエスト例（cURL）：
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "どう思いますか？"}'
```

レスポンス例：
```json
{
  "response": "それって〜じゃないですか？",
  "input": "どう思いますか？"
}
```

### Python から利用

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "こんにちは"}
)
print(response.json())
```

## 技術仕様

| 項目 | 仕様 |
|------|------|
| ベースモデル | Qwen/Qwen2.5-0.5B-Instruct |
| アダプタータイプ | LoRA |
| 量子化 | 4-bit (bitsandbytes) |
| 最大シーケンス長 | 2048 |
| 生成トークン数 | 最大 128 |
| サンプリング設定 | temperature=0.75, top_p=0.9, repetition_penalty=1.1 |
| フレームワーク | FastAPI + Uvicorn |
| 学習ライブラリ | Unsloth + PEFT + TRL |

## ひろゆき風スタイルの特徴

【基本スタイル】
- 冷静で論理的に話す
- 相手の前提や主張を疑う
- 「〜だと思うんですけど」「〜じゃないですかね」を多用
- 少し皮肉やズレた視点を混ぜる

【よく使う言い回し】
- 「それって〜じゃないですか？」
- 「なんか勘違いしてると思うんですけど」
- 「いや、普通に考えて」
- 「〜する意味あります？」
- 「別に〜すればよくないですか？」

## 開発者向け情報

### モデルのカスタマイズ

`slm_model.py` の `HIROYUKI_SYSTEM_PROMPT` を変更することで、応答のトーンを調整できます。

### 再学習

新しいデータで学習する場合は、TRL と Unsloth を使用して LoRA アダプターを再学習できます。

## ライセンス

MIT License

## 謝辞

このプロジェクトは以下のオープンソースプロジェクトに依存しています：

- [Unsloth](https://github.com/unslothai/unsloth) - 高速な LLM ファインチューニング
- [PEFT](https://github.com/huggingface/peft) - パラメーター効率的ファインチューニング
- [Qwen2.5](https://huggingface.co/Qwen) - ベースモデル
- [FastAPI](https://fastapi.tiangolo.com/) - API フレームワーク

---

**注意**: このモデルはひろゆきさんの話し方を模倣するものであり、実際のひろゆきさん本人とは関係ありません。
