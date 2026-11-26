"""§3.6 RAG correctness checks, on the real index and test questions.

Run from the repo root after index.py:

    python part_d_rag/checks.py

  - every chunk has at most 150 words
  - every chunk embedding, and every question embedding, has norm 1
  - with the relevant ids placed first (a perfect retriever), the metric code
    returns Hit@k = Recall@k = MRR@10 = 1
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(HERE))

from bioasq import load_corpus, load_questions
from chunking import CHUNK_WORDS
from encoder import Encoder
from index import load_index
from metrics import retrieval_report


def main():
    embeddings, texts, ids = load_index()
    longest = max(len(t.split()) for t in texts)
    print(f"chunks: {len(texts)}  longest chunk: {longest} words (limit {CHUNK_WORDS})")
    assert longest <= CHUNK_WORDS

    norms = embeddings.norm(dim=1)
    print(f"chunk embedding norms: min {norms.min():.6f}  max {norms.max():.6f}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    corpus = load_corpus()
    questions = load_questions("test", corpus)
    q_norms = Encoder().encode([q["question"] for q in questions]).norm(dim=1)
    print(f"question embedding norms: min {q_norms.min():.6f}  max {q_norms.max():.6f}")
    assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-4)

    all_ids = sorted(corpus)
    perfect = []
    for q in questions:
        rel = set(q["relevant_ids"])
        others = [a for a in all_ids[:40] if a not in rel][:20]
        perfect.append(list(q["relevant_ids"]) + others)
    report, curve = retrieval_report(perfect, [q["relevant_ids"] for q in questions])
    print("perfect retriever:", "  ".join(f"{k} {v:.4f}" for k, v in report.items()))
    assert all(abs(v - 1) < 1e-12 for v in report.values())
    assert all(abs(v - 1) < 1e-12 for v in curve)
    print("all RAG checks passed")


if __name__ == "__main__":
    main()
