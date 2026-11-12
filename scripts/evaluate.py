"""Compare retrieval quality and speed with different reranking settings.
Usage: python -m scripts.evaluate --k 5
"""
import argparse
import json
import statistics
from pathlib import Path

from app.retrieval import search

EVAL_PATH = Path("data/eval/eval_set.json")


def is_relevant(hit: dict, item: dict) -> bool:
    if hit["source"] != item["source"]:
        return False
    snippet = item.get("snippet")
    return not snippet or snippet.lower() in hit["text"].lower()


def run(items: list[dict], k: int, rerank: bool, candidates: int) -> dict:
    hits, rr, lat, misses = 0, [], [], []
    for item in items:
        results, ms = search(
            item["question"], k=k, rerank=rerank, candidates=candidates
        )
        lat.append(ms)
        rank = next((i + 1 for i, h in enumerate(results) if is_relevant(h, item)), None)
        if rank:
            hits += 1
            rr.append(1 / rank)
        else:
            rr.append(0.0)
            misses.append(item["question"])
    n = len(items)
    return {
        "hit": hits / n,
        "mrr": statistics.mean(rr),
        "p50": statistics.median(lat),
        "misses": misses,
    }


def main(k: int) -> None:
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    print("Loading models...")
    search("warm up", k=1, rerank=True)

    configs = [
        ("No rerank", False, k),
        ("Rerank 10", True, 10),
        ("Rerank 20", True, 20),
        ("Rerank 30", True, 30),
    ]
    results = []
    for name, rerank, cand in configs:
        print(f"Running: {name} ...")
        results.append((name, run(items, k, rerank, cand)))

    print(f"\nQuestions: {len(items)}\n")
    print(f"{'Setting':<14}{'Hit@' + str(k):>8}{'MRR':>8}{'p50 latency':>14}")
    for name, r in results:
        print(f"{name:<14}{r['hit']:>8.3f}{r['mrr']:>8.3f}{r['p50']:>11.0f} ms")

    for name, r in results:
        if r["misses"]:
            print(f"\nMissed ({name}):")
            for q in r["misses"]:
                print(f"  - {q}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    main(ap.parse_args().k)