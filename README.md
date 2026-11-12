# Hybrid Search over Research Papers

A retrieval system that finds the most relevant passages from a collection of research papers. It combines dense (meaning-based) and BM25 (keyword-based) search, then reranks the candidates with a cross-encoder. An evaluation script measures retrieval quality and speed.

## Features
- Hybrid search: dense embeddings + BM25 sparse vectors, fused with Reciprocal Rank Fusion (RRF)
- Optional cross-encoder reranking
- Metadata filtering by source file
- FastAPI `/search` endpoint with built-in test page at `/docs`
- Evaluation script reporting Hit@k, MRR and latency

## Tech stack
Python, FastAPI, Qdrant (Docker), FastEmbed (ONNX), pypdf, python-docx

## Project structure
```
voicerag/
├── app/
│   ├── config.py       # settings (database URL, chunk size, models)
│   ├── ingest.py       # reads PDFs, chunks text, embeds, stores in Qdrant
│   ├── retrieval.py    # hybrid search + reranking
│   └── main.py         # FastAPI server
├── scripts/
│   └── evaluate.py     # measures Hit@k, MRR, latency
├── data/
│   ├── docs/           # source documents (PDF, DOCX, TXT, MD)
│   └── eval/
│       └── eval_set.json   # test questions with expected sources
├── docker-compose.yml  # runs Qdrant
└── requirements.txt
```

## How it works
1. **Ingest:** each document is split into overlapping chunks (about 250 words, 40-word overlap). Every chunk gets a dense vector and a sparse BM25 vector, stored in Qdrant with metadata (source file, page, chunk id).
2. **Search:** a query is embedded both ways. Qdrant runs both searches and merges the rankings with RRF.
3. **Rerank (optional):** a cross-encoder reads the question together with each candidate passage and re-scores them. The best k are returned.
4. **Evaluate:** `scripts/evaluate.py` runs 14 test questions and checks whether a passage from the correct source, containing the expected phrase, appears in the top k.

## Setup and run
Requires Python 3.12 or 3.13 and Docker Desktop.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
docker compose up -d
```

Put your documents in `data/docs/`, then:

```powershell
python -m app.ingest --recreate
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs and try `GET /search`.

| Parameter | Meaning |
|---|---|
| `q` | Your question or keywords |
| `k` | Number of passages to return (default 5) |
| `source` | Optional. Search only one file, e.g. `bert.pdf` |

## Evaluation
In a second terminal (with the venv active):

```powershell
python -m scripts.evaluate --k 5
```

- **Hit@k:** share of questions where a correct passage appears in the top k results
- **MRR:** how high the first correct passage ranks (1.0 = always first)
- **p50 latency:** typical time for one search

## Results
Test set: 14 questions over 5 papers (Attention Is All You Need, RAG, BERT, GPT-3, Llama 2). CPU only, k = 5.

| Setting | Hit@5 | MRR | p50 latency |
|---|---|---|---|
| No rerank | 0.714 | 0.589 | 278 ms |
| Rerank 10 | 0.857 | 0.595 | 1993 ms |
| Rerank 20 | 0.929 | 0.588 | 3340 ms |
| Rerank 30 | 0.929 | 0.588 | 4323 ms |

**Findings**
- Reranking 20 candidates recovered 3 of the 4 questions missed by plain hybrid search.
- A pool of 30 added latency but no accuracy.
- MRR did not improve, so reranking helped coverage more than ranking order.
- Reranking is slow on CPU, which makes it a poor fit for real-time use without further optimization.
- Known failure: a plain-language question about positional encoding was missed in every setting, because its wording differs from the paper's ("sine and cosine functions").
- Latency numbers varied between runs on the same machine, so compare rows within one run.

## Limitations
- Returns passages only. It does not generate answers.
- Scanned (image-only) PDFs are not supported, since there is no OCR.
- The test set is small (14 questions), so results are indicative, not statistically strong.

## Future work
- LLM answer generation with citations and a "not found" fallback
- Faster reranking (smaller chunks or a lighter model)
- Larger evaluation set
- Optional voice interface