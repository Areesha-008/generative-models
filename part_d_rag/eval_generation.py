"""D.5–D.6 Generate answers for the first 100 test questions and evaluate them.

Run from the repo root after index.py and eval_retrieval.py:

    python part_d_rag/eval_generation.py

Writes to results/:
  rag_answers.jsonl                 one record per question: answer, passages, metrics
  part_d_generation.txt             Table 5: ROUGE-L, citation precision, "I don't know" rate
  part_d_retrieval_failures.md      three questions where retrieval failed, with answers (E.6)
  samples.md                        five questions for the blog (4.7)
"""
import json
import sys
from pathlib import Path

from rouge_score import rouge_scorer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(HERE))

from bioasq import load_corpus, load_questions
from encoder import Encoder
from generator import MEDICAL_NOTICE, TOP_CHUNKS, Generator
from index import load_index
from metrics import citation_counts, cited_numbers, is_idk
from search import Retriever

RESULTS = ROOT / "results"
RANKINGS = RESULTS / "part_d_rankings.jsonl"
N_QUESTIONS = 100
N_FAILURES = 3


def run(questions, encoder, retriever, generator, scorer, hit10):
    vecs = encoder.encode([q["question"] for q in questions])
    passages = [retriever.top_chunks(v, k=TOP_CHUNKS) for v in vecs]
    answers = generator.answer([q["question"] for q in questions], passages)
    records = []
    for q, ps, answer in zip(questions, passages, answers):
        abstract_ids = [p["abstract_id"] for p in ps]
        correct, total = citation_counts(answer, abstract_ids, q["relevant_ids"])
        records.append({
            "qid": q["qid"],
            "question": q["question"],
            "reference": q["answer"],
            "answer": answer,
            "passages": [{"n": i, **p} for i, p in enumerate(ps, start=1)],
            "relevant_ids": q["relevant_ids"],
            "retrieval_hit@10": hit10[q["qid"]],
            "cited": cited_numbers(answer),
            "citations_correct": correct,
            "citations_total": total,
            "idk": is_idk(answer),
            "rougeL": scorer.score(q["answer"], answer)["rougeL"].fmeasure,
        })
    return records


def describe(r, all_passages=False):
    """Markdown block for one answered question."""
    rel = set(r["relevant_ids"])
    lines = [
        f"### Q{r['qid']}: {r['question']}",
        "",
        f"**Answer:** {r['answer']}",
        "",
        f"*{MEDICAL_NOTICE}*",
        "",
        f"**Reference answer:** {r['reference']}",
        "",
        f"Retrieval hit@10: {'yes' if r['retrieval_hit@10'] else 'no'} · "
        f"said \"I don't know\": {'yes' if r['idk'] else 'no'} · "
        f"cited: {r['cited'] or 'none'} · ROUGE-L {r['rougeL']:.3f} · "
        f"relevant abstracts: {r['relevant_ids']}",
        "",
    ]
    shown = r["passages"] if all_passages else [p for p in r["passages"] if p["n"] in r["cited"]]
    lines.append("**Passages:**" if all_passages else "**Cited passages:**")
    lines.append("")
    if not shown:
        lines.append("_none cited_")
    for p in shown:
        mark = "relevant" if p["abstract_id"] in rel else "not relevant"
        lines.append(f"- [{p['n']}] abstract {p['abstract_id']} ({mark}, score {p['score']:.3f}): {p['text']}")
    lines.append("")
    return "\n".join(lines)


def main():
    if not RANKINGS.is_file():
        raise SystemExit(f"Missing {RANKINGS}. Run part_d_rag/eval_retrieval.py first.")
    with RANKINGS.open(encoding="utf-8") as f:
        hit10 = {row["qid"]: row["hit@10"] for row in map(json.loads, f)}

    corpus = load_corpus()
    questions = load_questions("test", corpus)
    embeddings, texts, ids = load_index()
    retriever = Retriever(embeddings, texts, ids)
    encoder = Encoder()
    generator = Generator()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    first = questions[:N_QUESTIONS]
    records = run(first, encoder, retriever, generator, scorer, hit10)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "rag_answers.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    rouge = sum(r["rougeL"] for r in records) / n
    cited_total = sum(r["citations_total"] for r in records)
    cited_correct = sum(r["citations_correct"] for r in records)
    precision = cited_correct / cited_total if cited_total else float("nan")
    idk_rate = sum(r["idk"] for r in records) / n
    no_citation = sum(1 for r in records if not r["cited"])
    summary = (
        f"questions: {n}\nrougeL_f: {rouge:.4f}\n"
        f"citation_precision: {precision:.4f} ({cited_correct}/{cited_total})\n"
        f"idk_rate: {idk_rate:.4f}\nanswers_without_citation: {no_citation}\n"
    )
    (RESULTS / "part_d_generation.txt").write_text(summary, encoding="utf-8")
    print(summary)

    # E.6: the first three questions (whole test set) whose top 10 has no relevant abstract.
    done = {r["qid"]: r for r in records}
    failed_qs = [q for q in questions if not hit10[q["qid"]]][:N_FAILURES]
    todo = [q for q in failed_qs if q["qid"] not in done]
    if todo:
        done.update({r["qid"]: r for r in run(todo, encoder, retriever, generator, scorer, hit10)})
    failures = [done[q["qid"]] for q in failed_qs]
    (RESULTS / "part_d_retrieval_failures.md").write_text(
        "# Part D: retrieval failures (no relevant abstract in the top 10)\n\n"
        + "\n".join(describe(r, all_passages=True) for r in failures),
        encoding="utf-8",
    )

    # 4.7: five samples, at least one retrieval failure and one "I don't know".
    picks = []
    failure = next((r for r in records if not r["retrieval_hit@10"]), failures[0] if failures else None)
    idk = next((r for r in records if r["idk"]), None)
    for r in (failure, idk):
        if r is not None and r not in picks:
            picks.append(r)
    for r in records:
        if len(picks) == 5:
            break
        if r not in picks and r["retrieval_hit@10"] and not r["idk"]:
            picks.append(r)
    notes = []
    if idk is None:
        notes.append("_None of the 100 answers said \"I don't know\"._\n")
    (RESULTS / "samples.md").write_text(
        "# Part D: qualitative samples\n\n" + "".join(notes)
        + "\n".join(describe(r) for r in picks),
        encoding="utf-8",
    )
    print("saved rag_answers.jsonl, part_d_generation.txt, part_d_retrieval_failures.md, samples.md")


if __name__ == "__main__":
    main()
