import asyncio
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_MODEL = "tmpai/Hiroyuki-SLM-LoRA"

load_dotenv()
USE_LORA = os.getenv("USE_LORA", "true").lower() == "true"

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
    def __init__(self) -> None:
        has_cuda = torch.cuda.is_available()
        print(f"CUDA available: {has_cuda}")

        device_map = "auto" if has_cuda else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map=device_map,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
        )

        if USE_LORA:
            self.model = PeftModel.from_pretrained(
                base_model,
                LORA_MODEL
            )
            self.model = self.model.merge_and_unload()
            print("Model + LoRA loaded successfully.")
        else:
            self.model = base_model
            print("Model loaded successfully (LoRA disabled).")

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": HIROYUKI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

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
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return response
