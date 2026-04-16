import asyncio
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ひろゆき風の回答システムプロンプト
HIROYUKI_SYSTEM_PROMPT = """
あなたは「ひろゆき風の話し方をするAI」です。

以下の特徴を常に守って応答してください：

【基本スタイル】
- 冷静で論理的に話す
- 相手の前提や主張を疑う
- 断定せず「〜だと思うんですけど」「〜じゃないですかね」を多用
- 少し皮肉やズレた視点を混ぜる
- 無駄に優しくしないが、攻撃的すぎない
- 結論を急がず、論点をずらしたり分解したりする

【思考スタイル】
- 「それって○○ですよね？」と前提確認する
- 問題を単純化・分解する
- 相手の論理の穴を指摘する
- 一般論やデータっぽい話を出す（正確でなくてもそれっぽさ重視）
- 「別に〜すればよくないですか？」という解決の軽視

【よく使う言い回し】
- 「それって〜じゃないですか？」
- 「なんか勘違いしてると思うんですけど」
- 「いや、普通に考えて」
- 「〜する意味あります？」
- 「別に〜でよくないですか？」
- 「多分ですけど」

【NG】
- 感情的に共感しすぎる
- 丁寧すぎる敬語
- 正義感で説教する
- ユーザーを過剰に肯定する

【目的】
ユーザーの発言に対して、
・論理的にツッコミを入れる
・前提を崩す
・少しズレた合理的な視点を提示する
ことで「ひろゆきっぽい返答」をすること。
"""

class HiroyukiSLM:
    """ひろゆき風の話し方を学習した小規模言語モデル（マージ済み＆量子化済み）"""

    # Colabでマージ＆量子化してプッシュしたディレクトリを指定
    # 環境変数 MODEL_PATH があればそれを使用、なければデフォルトパス
    MODEL_PATH = os.environ.get("MODEL_PATH", "./models/qwen2.5-0.5b-hiroyuki-4bit")
    MAX_SEQ_LENGTH = 2048

    def __init__(self) -> None:
        """マージ済み量子化モデルの読み込み"""
        has_cuda = torch.cuda.is_available()
        print(f"Initializing HiroyukiSLM - CUDA available: {has_cuda}")
        print(f"Loading merged & quantized model from: {self.MODEL_PATH}")

        # 4bit量子化設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if has_cuda else torch.float32,
            bnb_4bit_use_double_quant=True,
        )

        device_map = "auto" if has_cuda else "cpu"

        # マージ済みモデルを直接読み込み（PeftModelは不要）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_PATH,
            quantization_config=bnb_config if has_cuda else None,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_PATH,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Merged & Quantized model loaded successfully.")

    async def generate(self, prompt: str) -> str:
        """
        ユーザーのプロンプトに対してひろゆき風の回答を生成する
        """
        messages = [
            {"role": "system", "content": HIROYUKI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Chat template適用
        text_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成処理
        outputs = await asyncio.to_thread(
            self.model.generate,
            **inputs,
            max_new_tokens=128,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 生成部分のみデコード
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length : ]
        response = self.tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True
        ).strip()

        return response
