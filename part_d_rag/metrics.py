"""§3.4–3.5 Retrieval and generation metrics.

ranked: abstract ids, best first. relevant: the question's relevant_ids.
"""
import re


def hit_at_k(ranked, relevant, k):
    """1 if at least one relevant abstract is in the top k, else 0."""
    return float(bool(set(ranked[:k]) & set(relevant)))


def recall_at_k(ranked, relevant, k):
    """Relevant abstracts found in the top k, out of as many as could fit: min(k, |R|)."""
    relevant = set(relevant)
    return len(set(ranked[:k]) & relevant) / min(k, len(relevant))


def mrr_at_10(ranked, relevant):
    """1 / rank of the first relevant abstract if it is in the top 10, else 0."""
    relevant = set(relevant)
    for rank, abstract_id in enumerate(ranked[:10], start=1):
        if abstract_id in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_report(rankings, relevants, ks=(1, 5, 10), curve_max=20):
    """Means over all questions. Returns a dict of the table metrics and the Recall@k curve."""
    n = len(rankings)
    report = {}
    for k in ks:
        report[f"hit@{k}"] = sum(hit_at_k(r, rel, k) for r, rel in zip(rankings, relevants)) / n
    for k in ks:
        report[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in zip(rankings, relevants)) / n
    report["mrr@10"] = sum(mrr_at_10(r, rel) for r, rel in zip(rankings, relevants)) / n
    curve = [
        sum(recall_at_k(r, rel, k) for r, rel in zip(rankings, relevants)) / n
        for k in range(1, curve_max + 1)
    ]
    return report, curve


IDK_PATTERNS = [
    r"\bi (?:do not|don't) know\b",
    r"\b(?:do|does) not contain\b",
    r"\bdon't contain\b",
    r"\bnot (?:contain|provide|include|mention)\w* (?:the |an |any |enough )?(?:answer|information)\b",
    r"\bcannot (?:be )?(?:answer|determine)",
]


def is_idk(answer):
    """True if the answer says the passages do not contain the answer."""
    text = answer.lower().replace("’", "'")
    return any(re.search(p, text) for p in IDK_PATTERNS)


def cited_numbers(answer):
    """Distinct passage numbers cited in square brackets: '[1]', '[2, 4]', '[1][3]'."""
    numbers = set()
    for group in re.findall(r"\[([\d,\s\-–]+)\]", answer):
        for part in re.split(r"[,\s]+", group.strip()):
            if re.fullmatch(r"\d+[-–]\d+", part):
                lo, hi = map(int, re.split(r"[-–]", part))
                numbers.update(range(lo, hi + 1))
            elif part.isdigit():
                numbers.add(int(part))
    return sorted(numbers)


def citation_counts(answer, passage_abstract_ids, relevant):
    """(correct, total) citations in one answer. A citation is correct when it
    points to a passage whose abstract is relevant; numbers outside 1..5 are wrong."""
    relevant = set(relevant)
    cited = cited_numbers(answer)
    correct = sum(
        1 for n in cited
        if 1 <= n <= len(passage_abstract_ids) and passage_abstract_ids[n - 1] in relevant
    )
    return correct, len(cited)


if __name__ == "__main__":
    # A perfect retriever (relevant ids first) scores 1 on every metric.
    rel = [5, 9, 2]
    perfect = rel + [100 + i for i in range(20)]
    for k in (1, 5, 10):
        assert hit_at_k(perfect, rel, k) == 1 and recall_at_k(perfect, rel, k) == 1
    assert mrr_at_10(perfect, rel) == 1
    report, curve = retrieval_report([perfect], [rel])
    assert all(v == 1 for v in report.values()) and all(v == 1 for v in curve)
    # Partial cases.
    ranked = [1, 9, 3, 4, 5]
    assert recall_at_k(ranked, rel, 1) == 0 and recall_at_k(ranked, rel, 5) == 2 / 3
    assert mrr_at_10(ranked, rel) == 0.5 and mrr_at_10(list(range(100, 120)), rel) == 0
    assert recall_at_k([5], [5], 10) == 1          # denominator is min(k, |R|), never above 1
    # Answers.
    assert cited_numbers("Yes [1], see [2, 4] and [3][5]. Also [1-2].") == [1, 2, 3, 4, 5]
    assert citation_counts("A [1] B [2] C [7]", [5, 8, 9, 9, 9], rel) == (1, 3)
    assert is_idk("I don't know. The passages do not contain the answer.")
    assert is_idk("The provided passages do not contain information about this.")
    assert not is_idk("Aspirin irreversibly inhibits COX-1 [1].")
    print("checks passed")
