"""D.1 Split abstracts into overlapping word chunks.

Each chunk has at most 150 words, and consecutive chunks of one abstract share
30 words, so a new chunk starts every 150 - 30 = 120 words.
"""
CHUNK_WORDS = 150
OVERLAP_WORDS = 30


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Return the chunks of one text as strings. Empty text gives no chunks."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    start = 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            return chunks
        start += step


def chunk_corpus(corpus):
    """corpus: {abstract_id: text}. Returns (texts, abstract_ids), one entry per chunk."""
    texts, ids = [], []
    for abstract_id in sorted(corpus):
        for chunk in chunk_words(corpus[abstract_id]):
            texts.append(chunk)
            ids.append(abstract_id)
    return texts, ids


if __name__ == "__main__":
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_words(" ".join(words))
    sizes = [len(c.split()) for c in chunks]
    assert sizes == [150, 150, 150, 40], sizes      # starts at 0, 120, 240, 360
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-OVERLAP_WORDS:] == b.split()[:OVERLAP_WORDS]
    assert chunk_words("short abstract") == ["short abstract"]
    assert chunk_words("   ") == []
    assert len(chunk_words(" ".join(words[:150]))) == 1
    texts, ids = chunk_corpus({7: " ".join(words), 3: "tiny"})
    assert ids == [3, 7, 7, 7, 7]
    print("checks passed")
