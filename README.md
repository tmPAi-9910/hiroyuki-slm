# Hiroyuki SLM

Ultra-lightweight 4-bit quantized Small Language Model (SLM) for Hiroyuki-style chat responses.

## プロジェクト概要

Hiroyuki SLMは、超軽量な4ビット量子化Small Language Modelです。Hiroyukiさんのような_unique_な語り口を学習したチャットbotです。Embeddedデバイスやリソース制約のある環境でも動作するように設計されています。

## 特徴

- **Ultra-lightweight**: メモリ使用量 <500MB、ストレージ <1GB、1コアだけで動作
- **4-bit Quantization**: 効率的な量子化技術により、省リソースで動作
- **N-gram Model**: Trigramベースの高效な言語モデル（~100Kパラメータ）
- **Exact Match + SLM Fallback**: 定義済みレスポンスとの完全一致を優先し、マッチしない場合はSLMで生成
- **RESTful API**: FlaskベースのシンプルなAPI提供
- **日本語特化**: 日本語テキストの処理に特化

## ファイル構造

```
hiroyuki-slm/
├── api.py              # Flask APIサービス（/chat, /health エンドポイント）
├── slm_model.py        # 4ビット量子化SLMモデル（N-gram実装）
├── quotes.json         # Hiroyuki名言・引用データセット
├── responces.json      # 完全一致トリガーレスポンス
├── test.py             # テストスクリプト
└── README.md           # このファイル
```

## 必要環境

- Python 3.8+
- Flask
- requests

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-repo/hiroyuki-slm.git
cd hiroyuki-slm

# 依存関係をインストール
pip install flask requests
```

## 使い方

### APIサーバーの起動

```bash
python api.py
```

デフォルトではポート8080で起動します。環境変数で変更可能です：

```bash
PORT=3000 python api.py
```

### APIエンドポイント

#### 1. Health Check
```
GET /health
```

応答例：
```json
{
  "status": "healthy",
  "model": "hiroyuki-slm-4bit",
  "version": "1.0.0"
}
```

#### 2. Chat
```
POST /chat
Content-Type: application/json

{
  "message": "あなたのメッセージ"
}
```

応答例：
```json
{
  "response": "生成されたレスポンス",
  "input": "あなたのメッセージ"
}
```

#### 3. Streaming Chat
```
POST /chat/stream
```

（現在の実装では通常のchatエンドポイントと同じ動作をします）

#### 4. Model Info
```
GET /models/info
```

応答例：
```json
{
  "model_name": "hiroyuki-slm-4bit",
  "quantization": "4-bit",
  "parameters": "~2M",
  "vocab_size": 120,
  "context_length": 32
}
```

### テストの実行

```bash
# APIサーバーを起動後に別ターミナルで実行
python test.py
```

またはカスタムURL指定：
```bash
python test.py http://localhost:3000
```

## 技術仕様

| 項目 | 仕様 |
|------|------|
| メモリ使用量 | < 500MB |
| ストレージ | < 1GB |
| CPU | 1コア |
| パラメータ数 | ~100K |
| 量子化 | 4-bit |
| コンテキスト長 | 32トークン |
| モデルタイプ | N-gram (Trigram) |

## 使用例

### cURL

```bash
# 健康チェック
curl http://localhost:8080/health

# チャット
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "こんにちは"}'

# モデル情報
curl http://localhost:8080/models/info
```

### Python

```python
import requests

# チャットリクエスト
response = requests.post(
    "http://localhost:8080/chat",
    json={"message": "どう思いますか？"}
)
print(response.json())
```

### Exact Match Examples

`responces.json`で定義されている完全一致パターンの例：

| 入力 | レスポンス |
|------|------------|
| 嘘 | "何だろう。噓つくのやめてもらっていいですか？" |
| データ | "データなんかねえよ" |
| 学校 | "学校でしか学べない価値ってなんだろう..." |
| 時計 | "正しい時間を知るには時計二つではダメなんすよね" |

## ライセンス

MIT License

## 謝辞

このモデルはHiroyukiさんの名言・引用データを基に学習されています。
