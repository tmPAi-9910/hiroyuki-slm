import asyncio
import torch
from peft import PeftModel

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
    """ひろゆき風の話し方を学習した小規模言語モデル"""

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    ADAPTER_PATH = "hiroyuki_adapter"
    MAX_SEQ_LENGTH = 2048

    def __init__(self) -> None:
        """モデルとトークナイザーの初期化"""
        has_cuda = torch.cuda.is_available()
        print(f"Initializing HiroyukiSLM - CUDA available: {has_cuda}")

        if has_cuda:
            try:
                from unsloth import FastLanguageModel
                print("Loading model with unsloth...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.MODEL_NAME,
                    max_seq_length=self.MAX_SEQ_LENGTH,
                    load_in_4bit=True,
                    device_map="auto",
                )
                model = PeftModel.from_pretrained(model, self.ADAPTER_PATH)
                FastLanguageModel.for_inference(model)
                self.model = model
                self.tokenizer = tokenizer
                print("Unsloth model loaded successfully.")
            except (ImportError, Exception) as e:
                print(f"Unsloth loading failed ({e}), falling back to transformers")
                self._load_with_transformers()
        else:
            self._load_with_transformers()

    def _load_with_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model with transformers (CPU fallback)...")

        if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, self.ADAPTER_PATH)

        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer
        print("Transformers model loaded successfully.")

    async def generate(self, prompt: str) -> str:
        """
        ユーザーのプロンプトに対してひろゆき風の回答を生成する

        Args:
            prompt: ユーザーの入力プロンプト

        Returns:
            生成されたテキスト応答
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
