"""D.4 Measure retrieval on every test question.

Run from the repo root after index.py:

    python part_d_rag/eval_retrieval.py

Writes to results/:
  part_d_retrieval.csv     Table 4: Hit@1/5/10, Recall@1/5/10, MRR@10
  part_d_recall_at_k.csv   Recall@k for k = 1..20, and part_d_recall_at_k.png
  part_d_data.txt          Table 1 rows for BioASQ
  part_d_rankings.jsonl    top-20 abstracts per question (used by eval_generation.py)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(HERE))

from bioasq import load_corpus, load_questions
from encoder import Encoder
from index import load_index
from metrics import hit_at_k, retrieval_report
from search import Retriever

RESULTS = ROOT / "results"
RANKINGS = RESULTS / "part_d_rankings.jsonl"
TOP = 20


def main():
    corpus = load_corpus()
    questions = load_questions("test", corpus)
    embeddings, texts, ids = load_index()
    retriever = Retriever(embeddings, texts, ids)
    encoder = Encoder()

    query_vecs = encoder.encode([q["question"] for q in questions])
    rankings = retriever.top_abstracts(query_vecs, k=TOP)
    relevants = [q["relevant_ids"] for q in questions]
    report, curve = retrieval_report(rankings, relevants, curve_max=TOP)

    RESULTS.mkdir(parents=True, exist_ok=True)
    keys = list(report)
    (RESULTS / "part_d_retrieval.csv").write_text(
        ",".join(keys) + "\n" + ",".join(f"{report[k]:.4f}" for k in keys) + "\n",
        encoding="utf-8",
    )
    (RESULTS / "part_d_recall_at_k.csv").write_text(
        "k,recall\n" + "".join(f"{k},{r:.4f}\n" for k, r in enumerate(curve, start=1)),
        encoding="utf-8",
    )
    mean_relevant = sum(len(r) for r in relevants) / len(relevants)
    (RESULTS / "part_d_data.txt").write_text(
        f"abstracts: {len(corpus)}\nchunks: {len(texts)}\ntest_questions: {len(questions)}\n"
        f"mean_relevant_per_question: {mean_relevant:.2f}\n",
        encoding="utf-8",
    )
    with RANKINGS.open("w", encoding="utf-8") as f:
        for q, ranked in zip(questions, rankings):
            f.write(json.dumps({
                "qid": q["qid"],
                "top_abstracts": ranked,
                "relevant_ids": q["relevant_ids"],
                "hit@10": hit_at_k(ranked, q["relevant_ids"], 10),
            }) + "\n")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, TOP + 1), curve, marker="o", markersize=3)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_xticks(range(1, TOP + 1, 2))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title("Part D retrieval: Recall@k on BioASQ test")
    fig.tight_layout()
    fig.savefig(RESULTS / "part_d_recall_at_k.png", dpi=120)
    plt.close(fig)

    failures = sum(1 for r, rel in zip(rankings, relevants) if not hit_at_k(r, rel, 10))
    print(f"test questions: {len(questions)}  chunks: {len(texts)}  mean relevant: {mean_relevant:.2f}")
    print("  ".join(f"{k} {v:.4f}" for k, v in report.items()))
    print(f"retrieval failures (no relevant abstract in top 10): {failures}")
    print("saved", RESULTS / "part_d_retrieval.csv", "and", RESULTS / "part_d_recall_at_k.png")


if __name__ == "__main__":
    main()
