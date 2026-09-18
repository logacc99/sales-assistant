"""End-to-end live hybrid retrieval verification against Amazon OpenSearch Serverless."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.retrieval.models import RetrievalQuery, SearchType
from src.retrieval.retriever import OpenSearchHybridRetriever


def main() -> None:
    print("=" * 60)
    print("STEP 1: Initializing OpenSearchHybridRetriever")
    print("=" * 60)
    cfg = get_config()
    print(f"Target OpenSearch Index: {cfg.opensearch_index_name}")
    print(f"Serverless Mode: {cfg.opensearch_is_serverless}")
    print(f"Service Name: {cfg.opensearch_service_name}")

    retriever = OpenSearchHybridRetriever()

    print("\n" + "=" * 60)
    print("STEP 2: Verifying AOSS Index Vitality")
    print("=" * 60)
    exists = retriever.client.indices.exists(index=retriever.index_name)
    print(f"Index '{retriever.index_name}' exists: {exists}")
    assert exists, f"Index {retriever.index_name} does not exist!"

    count_res = retriever.client.count(index=retriever.index_name)
    doc_count = count_res.get("count", 0)
    print(f"Current Document Count: {doc_count}")

    print("\n" + "=" * 60)
    print("STEP 3: Executing Hybrid Search (BM25 + k-NN + RRF + Rerank)")
    print("=" * 60)
    test_query = "bep nuong gas ngoai troi"
    print(f"Query: '{test_query}'")

    result = retriever.retrieve(
        RetrievalQuery(
            query=test_query,
            search_type=SearchType.HYBRID,
            top_k=3,
        )
    )

    print(f"\nRetrieval Result:")
    print(f"- Total Hits: {result.total_hits}")
    print(f"- Latency: {result.latency_ms:.2f} ms")
    print(f"- Mode Used: {result.retrieval_mode_used}")
    print(f"- Degraded: {result.degraded} (Reason: {result.degradation_reason})")

    for i, chunk in enumerate(result.chunks, 1):
        print(f"\n[Rank {i}] (Score: {chunk.score:.4f}, Category: {chunk.category})")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Product Name: {chunk.metadata.get('product_name', 'N/A')}")
        print(f"Content: {chunk.content[:150]}...")

    print("\n" + "=" * 60)
    print("SUCCESS: Live OpenSearch Serverless retrieval verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
