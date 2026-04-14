import asyncio

from unsolth import FastLanguageModel
from peft import PeftModel


# ひろゆき風の回答システムプロンプト
HIROYUKI_SYSTEM_PROMPT = """\
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


class HiroyukiSLM(FastLanguageModel):
    """ひろゆき風の話し方を学習した小規模言語モデル"""

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    ADAPTER_PATH = "hiroyuki_adapter"
    MAX_SEQ_LENGTH = 2048

    def __init__(self) -> None:
        """モデルとトークナイザーの初期化"""
        super().__init__()

        # ベースモデルとトークナイザーをロード
        model, tokenizer = self.from_pretrained(
            model_name=self.MODEL_NAME,
            max_seq_length=self.MAX_SEQ_LENGTH,
            load_in_4bit=True,
            device_map="auto",
        )

        # アダプタを適用
        model = PeftModel.from_pretrained(model, self.ADAPTER_PATH)

        # 推論モード に設定
        self.for_inference(model)

        self.model = model
        self.tokenizer = tokenizer

    async def generate(self, prompt: str) -> str:
        """
        ユーザーのプロンプトに対してひろゆき風の回答を生成する

        Args:
            prompt: ユーザーの入力プロンプト

        Returns:
            生成されたテキスト応答
        """

        # メッセージの構築
        messages = [
            {"role": "system", "content": HIROYUKI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # トークン化
        inputs = self.tokenizer(messages, return_tensors="pt").to(self.model.device)

        # テキスト生成（重い処理をスレッドで実行）
        outputs = await asyncio.to_thread(
            self.model.generate,
            **inputs,
            max_new_tokens=128,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

        # 出力をデコード（スレッドで実行）
        decoded = await asyncio.to_thread(
            self.tokenizer.decode, outputs[0], True
        )

        # アシスタントの応答部分を抽出
        response = decoded.split("assistant:")[-1].strip()

        return response
