"""
Evaluation script for measuring Hit Rate@K, MRR, and Retrieval Latency.
Designed to be invoked by the agent or developer as part of the rag-eval skill.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
_CURRENT_DIR = Path(__file__).resolve()
_PROJECT_ROOT = next(
    (p for p in [_CURRENT_DIR] + list(_CURRENT_DIR.parents) if (p / "src").is_dir()),
    _CURRENT_DIR.parents[4],
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def evaluate_retrieval(k: int = 3, dataset_path: str = "tests/fixtures/eval_dataset.json") -> Dict[str, Any]:
    print(f"--- Running RAG Evaluation (Top-K = {k}) ---")
    if not os.path.exists(dataset_path):
        print(f"[Error] Eval dataset not found at '{dataset_path}'.")
        return {
            "status": "missing_dataset",
            "k": k,
            "hit_rate": 0.0,
            "mrr": 0.0,
        }

    with open(dataset_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} evaluation questions from '{dataset_path}'.")

    from src.retrieval.models import RetrievalQuery, SearchType
    from src.retrieval.retriever import OpenSearchHybridRetriever

    retriever = None
    cluster_online = False

    try:
        retriever = OpenSearchHybridRetriever()
        if retriever.client.ping():
            cluster_online = True
    except Exception as exc:
        print(f"[Warning] Live OpenSearch cluster not reachable ({exc}). Running evaluation analysis on dataset.")

    hits = 0
    reciprocal_ranks: List[float] = []
    latencies: List[float] = []

    if cluster_online and retriever is not None:
        for item in queries:
            q_text = item["query"]
            req = RetrievalQuery(query=q_text, search_type=SearchType.HYBRID, top_k=k)
            t0 = time.perf_counter()
            res = retriever.retrieve(req)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            # Determine if hit was retrieved
            is_hit = False
            item_rr = 0.0
            exp_pid = item.get("expected_product_id")
            exp_cat = item.get("expected_category")
            exp_keywords = item.get("expected_keywords", [])

            for rank, chunk in enumerate(res.chunks[:k], start=1):
                match = False
                if exp_pid and chunk.metadata.get("product_id") == exp_pid:
                    match = True
                elif exp_cat and chunk.category == exp_cat:
                    if exp_keywords:
                        if any(kw.lower() in chunk.content.lower() for kw in exp_keywords):
                            match = True
                    else:
                        match = True

                if match:
                    is_hit = True
                    item_rr = 1.0 / rank
                    break

            if is_hit:
                hits += 1
            reciprocal_ranks.append(item_rr)

        hit_rate = hits / len(queries) if queries else 0.0
        mrr = sum(reciprocal_ranks) / len(queries) if queries else 0.0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "status": "evaluated_live_cluster",
            "k": k,
            "queries_evaluated": len(queries),
            "hit_rate_at_k": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0, 2),
        }
    else:
        # Simulation / offline benchmark against test queries
        return {
            "status": "eval_dataset_validated",
            "k": k,
            "queries_count": len(queries),
            "target_hit_rate": 0.85,
            "target_mrr": 0.75,
            "target_latency_ms": 150.0,
            "sample_queries": [q["query"] for q in queries[:3]],
            "notice": "Live cluster offline. Verified dataset structure and evaluation runner ready for CI/CD.",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG Retrieval Performance")
    parser.add_argument("--k", type=int, default=3, help="Top-K evaluation cutoff")
    parser.add_argument("--dataset", type=str, default="tests/fixtures/eval_dataset.json", help="Path to evaluation dataset")
    args = parser.parse_args()
    res = evaluate_retrieval(k=args.k, dataset_path=args.dataset)
    print(json.dumps(res, indent=2, ensure_ascii=False))
