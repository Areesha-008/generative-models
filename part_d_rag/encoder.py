"""D.2 Sentence encoder: all-MiniLM-L6-v2 through transformers, with mean pooling.

The last hidden states are averaged over the real tokens only (the attention
mask zeroes out padding), then L2-normalised so cosine similarity is a dot
product. Questions and chunks are encoded the same way.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256
DIM = 384


def mean_pool(hidden, attention_mask):
    """hidden (N, L, D), attention_mask (N, L) -> (N, D), padding ignored."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class Encoder:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
        self.model = AutoModel.from_pretrained(ENCODER_NAME).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=256, show_progress=False):
        """Returns a float32 tensor (len(texts), 384) of unit vectors, on the CPU."""
        # Encode in length order so each batch has little padding, then restore order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), DIM)
        for n, start in enumerate(range(0, len(texts), batch_size)):
            idx = order[start:start + batch_size]
            batch = self.tokenizer(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            pooled = mean_pool(hidden, batch["attention_mask"])
            out[idx] = F.normalize(pooled, dim=-1).float().cpu()
            if show_progress and n % 50 == 0:
                print(f"  encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return out


if __name__ == "__main__":
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))
    enc = Encoder()
    v = enc.encode(["Aspirin inhibits platelet aggregation.", "a", "How does aspirin work?"])
    assert v.shape == (3, DIM)
    assert torch.allclose(v.norm(dim=1), torch.ones(3), atol=1e-5)
    # Padding must not change an embedding: encode alone vs. in a padded batch.
    alone = enc.encode(["a"])
    assert torch.allclose(alone[0], v[1], atol=1e-5)
    print("checks passed")
    print("cos(statement, question):", float(v[0] @ v[2]), " cos(statement, 'a'):", float(v[0] @ v[1]))
