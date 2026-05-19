"""Simple overlapping character chunker for parsed PDF documents."""


def chunk_documents(docs, chunk_size=512, overlap=64) -> list[dict]:
    """Split parsed documents into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks = []
    step = chunk_size - overlap

    for doc_index, doc in enumerate(docs):
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        source = doc.get("source", "unknown")
        page = doc.get("page", 0)
        parent_id = f"{source}:page-{page}"

        chunk_index = 0
        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()
            if not chunk_text:
                continue

            chunk = {
                "text": chunk_text,
                "chunk_id": f"{parent_id}:doc-{doc_index}:chunk-{chunk_index}",
                "parent_id": parent_id,
                "page": page,
                "source": source,
                "chunk_index": chunk_index,
            }
            if doc.get("type"):
                chunk["type"] = doc["type"]

            chunks.append(chunk)
            chunk_index += 1

            if start + chunk_size >= len(text):
                break

    return chunks

