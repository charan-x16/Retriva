"""In-memory BM25 retriever for indexed document chunks."""

from rank_bm25 import BM25Okapi

BM25_INDEX = None
BM25_CHUNKS = []


def build_bm25_index(chunks):
    """Build and cache a BM25 index over chunk text."""

    global BM25_INDEX, BM25_CHUNKS

    BM25_CHUNKS = list(chunks)
    tokenized_chunks = [_tokenize(chunk["text"]) for chunk in BM25_CHUNKS]
    BM25_INDEX = BM25Okapi(tokenized_chunks) if tokenized_chunks else None
    return BM25_INDEX, BM25_CHUNKS


def retrieve_bm25(query, top_k=20) -> list[tuple[dict, float]]:
    """Return top BM25 chunks and scores for a query."""

    if BM25_INDEX is None or not BM25_CHUNKS:
        return []

    scores = BM25_INDEX.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [(BM25_CHUNKS[index], float(score)) for index, score in ranked[:top_k]]


def _tokenize(text) -> list[str]:
    """Tokenize text for BM25 with a simple lowercase split."""

    return text.lower().split()

