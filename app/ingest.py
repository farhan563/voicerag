"""Parse -> chunk -> embed -> store.
Usage: python -m app.ingest --recreate
"""
import argparse
import uuid
from pathlib import Path

from docx import Document
from pypdf import PdfReader
from qdrant_client import models

from app import config
from app.retrieval import ensure_collection, get_client, get_dense, get_sparse

DOCS_DIR = Path("data/docs")
BATCH = 32  # chunks processed per round (lower this if you run out of RAM)


def read_file(path: Path) -> list[tuple[int | None, str]]:
    """Return a list of (page_number_or_None, text)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if ext == ".docx":
        doc = Document(str(path))
        return [(None, "\n".join(p.text for p in doc.paragraphs))]
    if ext in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    return []


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def main(recreate: bool) -> None:
    ensure_collection(recreate=recreate)
    dense, sparse, client = get_dense(), get_sparse(), get_client()

    records = []
    for path in sorted(DOCS_DIR.glob("*")):
        for page, text in read_file(path):
            for chunk in chunk_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP):
                records.append({"source": path.name, "page": page, "text": chunk})

    if not records:
        print(f"No documents found in {DOCS_DIR}. Add PDF/DOCX/TXT/MD files first.")
        return

    print(f"Embedding {len(records)} chunks...")
    total = 0
    for start in range(0, len(records), BATCH):
        batch = records[start:start + BATCH]
        texts = [r["text"] for r in batch]
        dense_vecs = list(dense.embed(texts, batch_size=16))
        sparse_vecs = list(sparse.embed(texts))

        points = []
        for offset, (r, d, s) in enumerate(zip(batch, dense_vecs, sparse_vecs)):
            chunk_id = start + offset
            r["chunk_id"] = chunk_id
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r['source']}:{r['page']}:{chunk_id}")),
                    vector={
                        "dense": d.tolist(),
                        "bm25": models.SparseVector(
                            indices=s.indices.tolist(), values=s.values.tolist()
                        ),
                    },
                    payload=r,
                )
            )
        client.upsert(config.COLLECTION, points=points)
        total += len(points)
        print(f"  stored {total}/{len(records)}")

    n_files = len({r["source"] for r in records})
    print(f"Done. Ingested {total} chunks from {n_files} files.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    main(ap.parse_args().recreate)