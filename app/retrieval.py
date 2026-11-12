"""Hybrid retrieval (dense + BM25 sparse, fused with RRF) plus cross-encoder reranking."""
import time
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from app import config

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


@lru_cache(maxsize=1)
def get_dense() -> TextEmbedding:
    return TextEmbedding(config.DENSE_MODEL)


@lru_cache(maxsize=1)
def get_sparse() -> SparseTextEmbedding:
    return SparseTextEmbedding(config.SPARSE_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANK_MODEL)


def ensure_collection(recreate: bool = False) -> None:
    client = get_client()
    exists = client.collection_exists(config.COLLECTION)
    if exists and recreate:
        client.delete_collection(config.COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=config.DENSE_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )


def search(
    query: str,
    k: int = 5,
    prefetch_limit: int = 20,
    source: str | None = None,
    rerank: bool = True,
    candidates: int = 20,
):
    """Return (results, latency_ms).

    1. Hybrid search fetches `candidates` passages (dense + BM25, fused with RRF).
    2. If rerank=True, a cross-encoder re-scores them and the best k are returned.
    """
    t0 = time.perf_counter()
    dense_vec = list(get_dense().query_embed(query))[0].tolist()
    sp = list(get_sparse().query_embed(query))[0]
    sparse_vec = models.SparseVector(
        indices=sp.indices.tolist(), values=sp.values.tolist()
    )

    flt = None
    if source:
        flt = models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
        )

    fetch = max(k, candidates) if rerank else k
    res = get_client().query_points(
        collection_name=config.COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit, filter=flt),
            models.Prefetch(query=sparse_vec, using="bm25", limit=prefetch_limit, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch,
        with_payload=True,
    )
    hits = [
        {
            "score": p.score,
            "text": p.payload["text"],
            "source": p.payload["source"],
            "chunk_id": p.payload["chunk_id"],
            "page": p.payload.get("page"),
        }
        for p in res.points
    ]

    if rerank and hits:
        scores = list(get_reranker().rerank(query, [h["text"] for h in hits], batch_size=8))
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        hits.sort(key=lambda h: h["score"], reverse=True)
        hits = hits[:k]

    latency_ms = (time.perf_counter() - t0) * 1000
    return hits, latency_ms