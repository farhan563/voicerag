from fastapi import FastAPI, Query

from app.retrieval import search

app = FastAPI(title="VoiceRAG")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search_endpoint(
    q: str = Query(..., description="Your question"),
    k: int = 5,
    source: str | None = None,
):
    hits, latency_ms = search(q, k=k, source=source)
    return {"query": q, "latency_ms": round(latency_ms, 1), "results": hits}