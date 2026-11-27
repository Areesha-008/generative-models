"""D.5 Answer a question from the top 5 retrieved chunks with Qwen2.5-1.5B-Instruct.

The passages are numbered [1]-[5]. The prompt tells the model to answer only
from them, cite passage numbers in square brackets, and say it does not know
when the passages do not contain the answer. Chat template, greedy decoding,
at most 256 new tokens, float16 on the GPU.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GENERATOR_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256
TOP_CHUNKS = 5
MEDICAL_NOTICE = "For educational use only. Not medical advice."

SYSTEM_PROMPT = (
    "You answer biomedical research questions using only the numbered passages "
    "you are given.\n"
    "Rules:\n"
    "1. Use only information from the passages, never your own knowledge.\n"
    "2. End every sentence with the number of the passage it comes from, in "
    "square brackets, for example: Drug X lowers LDL cholesterol [2]. Use [1][3] "
    "when a sentence draws on two passages.\n"
    "3. If the passages do not contain the answer, reply exactly: "
    "\"I don't know. The passages do not contain the answer.\"\n"
    "4. Answer in a few sentences."
)


def build_messages(question, passages):
    """passages: list of dicts with 'text'. Returns chat messages for the template."""
    numbered = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))
    # Small models follow the instruction closest to the end best, so the
    # citation rule is repeated after the question.
    user = (
        f"Passages:\n\n{numbered}\n\nQuestion: {question}\n\n"
        "Answer in a few sentences using only the passages above. After every "
        "sentence, write the number of the passage it comes from in square "
        "brackets, like [1] or [2][4]. If the passages do not contain the answer, "
        "reply exactly: \"I don't know. The passages do not contain the answer.\""
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(GENERATOR_NAME, dtype=dtype)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def answer(self, questions, passages_per_question, batch_size=16):
        """Greedy answers, one per question. Batched with left padding."""
        prompts = [
            self.tokenizer.apply_chat_template(
                build_messages(q, ps), tokenize=False, add_generation_prompt=True
            )
            for q, ps in zip(questions, passages_per_question)
        ]
        answers = []
        for start in range(0, len(prompts), batch_size):
            batch = self.tokenizer(
                prompts[start:start + batch_size],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)
            out = self.model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = out[:, batch["input_ids"].shape[1]:]
            answers.extend(
                t.strip() for t in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
        return answers


if __name__ == "__main__":
    passages = [{"text": "Aspirin irreversibly inhibits cyclooxygenase-1 in platelets."}]
    messages = build_messages("What does aspirin inhibit?", passages)
    assert "[1] Aspirin" in messages[1]["content"]
    assert "I don't know" in messages[0]["content"]
    print("prompt checks passed")
    gen = Generator()
    print(gen.answer(["What does aspirin inhibit?", "What is the capital of France?"],
                     [passages, passages]))
