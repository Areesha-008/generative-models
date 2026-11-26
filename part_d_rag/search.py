"""D.3 Search the chunk index.

Every vector is unit length, so cosine similarity is a dot product and the
scores of a batch of questions against all chunks are one matrix product.
An abstract's score is the highest score of its chunks.
"""
import torch


class Retriever:
    def __init__(self, embeddings, chunk_texts, chunk_abstract_ids, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb = embeddings.to(self.device)
        self.texts = chunk_texts
        self.chunk_abstract_ids = chunk_abstract_ids
        # Map each chunk to a column 0..A-1, one column per abstract.
        self.abstract_ids = sorted(set(chunk_abstract_ids))
        column = {a: i for i, a in enumerate(self.abstract_ids)}
        self.chunk_col = torch.tensor(
            [column[a] for a in chunk_abstract_ids], device=self.device
        )

    def chunk_scores(self, query_vecs):
        """(Q, 384) unit vectors -> (Q, n_chunks) cosine similarities."""
        return query_vecs.to(self.device) @ self.emb.T

    def abstract_scores(self, query_vecs):
        """(Q, n_abstracts): for each abstract, the max over its chunks."""
        scores = self.chunk_scores(query_vecs)
        out = torch.full(
            (scores.shape[0], len(self.abstract_ids)), float("-inf"), device=self.device
        )
        index = self.chunk_col.unsqueeze(0).expand_as(scores)
        return out.scatter_reduce(1, index, scores, reduce="amax", include_self=True)

    def top_abstracts(self, query_vecs, k=10, batch_size=128):
        """List (one per query) of the top-k abstract ids, best first."""
        results = []
        for start in range(0, len(query_vecs), batch_size):
            scores = self.abstract_scores(query_vecs[start:start + batch_size])
            top = scores.topk(k, dim=1).indices.cpu().tolist()
            results.extend([[self.abstract_ids[c] for c in row] for row in top])
        return results

    def top_chunks(self, query_vec, k=5):
        """Top-k chunks for one query: list of dicts with text, abstract_id, score."""
        scores = self.chunk_scores(query_vec.view(1, -1))[0]
        top = scores.topk(k)
        return [
            {"text": self.texts[i], "abstract_id": self.chunk_abstract_ids[i], "score": float(s)}
            for s, i in zip(top.values.cpu().tolist(), top.indices.cpu().tolist())
        ]


if __name__ == "__main__":
    import torch.nn.functional as F

    emb = F.normalize(torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [0.8, 0.6]]), dim=1)
    r = Retriever(emb, ["a0", "a1", "b0", "c0"], [10, 10, 20, 30], device=torch.device("cpu"))
    q = F.normalize(torch.tensor([[0.0, 1.0]]), dim=1)
    # Abstract 10 scores max(0, 0.8) = 0.8; 20 scores 1.0; 30 scores 0.6.
    assert torch.allclose(r.abstract_scores(q), torch.tensor([[0.8, 1.0, 0.6]]))
    assert r.top_abstracts(q, k=3) == [[20, 10, 30]]
    assert [c["text"] for c in r.top_chunks(q[0], k=2)] == ["b0", "a1"]
    print("checks passed")
