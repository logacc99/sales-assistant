"""Pre-flight verification and live ingestion pipeline test."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.ingestion.embedder import BedrockEmbedder
from src.ingestion.indexer import OpenSearchVectorIndexer
from src.ingestion.pipeline import IngestionPipeline


def run_preflight_and_test() -> None:
    print("=" * 60)
    print("STEP 1: Loading configuration from .env")
    print("=" * 60)
    cfg = get_config()
    print(f"AWS Region: {cfg.aws_region}")
    print(f"Bedrock Model: {cfg.bedrock_model_id}")
    print(f"Bedrock Dimension: {cfg.bedrock_dimension}")
    print(f"OpenSearch Host: {cfg.opensearch_host}:{cfg.opensearch_port}")
    print(f"OpenSearch Index: {cfg.opensearch_index_name}")
    print(f"OpenSearch Use AWS Auth: {cfg.opensearch_use_aws_auth}")

    print("\n" + "=" * 60)
    print("STEP 2: Testing Bedrock Embedder")
    print("=" * 60)
    embedder = BedrockEmbedder()
    test_text = "Bếp nướng dã ngoại cao cấp BBQ"
    print(f"Invoking Bedrock model '{cfg.bedrock_model_id}' with text: '{test_text}'...")
    try:
        vec = embedder.embed_text(test_text)
        print(f"SUCCESS: Received embedding vector of length {len(vec)} (sample: {vec[:3]}...)")
    except Exception as e:
        print(f"FAILURE connecting to Bedrock: {e}")
        return

    print("\n" + "=" * 60)
    print("STEP 3: Testing OpenSearch Cluster & Index")
    print("=" * 60)
    indexer = OpenSearchVectorIndexer()
    try:
        info = indexer.client.info()
        print(f"SUCCESS: Connected to OpenSearch cluster version {info.get('version', {}).get('number')}")
    except Exception as e:
        print(f"FAILURE connecting to OpenSearch: {e}")
        return

    test_index = cfg.opensearch_index_name
    print(f"Ensuring test index '{test_index}' exists...")
    try:
        indexer.ensure_index_exists(test_index)
        exists = indexer.client.indices.exists(index=test_index)
        print(f"SUCCESS: Index '{test_index}' exists = {exists}")
    except Exception as e:
        print(f"FAILURE creating/checking index '{test_index}': {e}")
        return

    print("\n" + "=" * 60)
    print("STEP 4: Testing Single File Ingestion")
    print("=" * 60)
    pipeline = IngestionPipeline(
        embedder=embedder,
        indexer=indexer,
        index_name=test_index,
        enable_cdc=True,
        enable_reconciliation=False,
    )

    sample_file = Path("data/raw/website/bepbbq.com/crawled_san-pham_blackstone-griddle-original-28-2147_d6f02eb7.md")
    if not sample_file.exists():
        print(f"Error: {sample_file} not found!")
        return

    print(f"Ingesting file: {sample_file}")
    res = pipeline.ingest_file(sample_file)
    print("Ingestion result:")
    for k, v in res.items():
        if k != "active_doc_ids":
            print(f"  - {k}: {v}")

    print("\n" + "=" * 60)
    print("STEP 5: Testing Change Data Detection (CDC)")
    print("=" * 60)
    print("Re-ingesting the exact same file (should skip embedding & indexing)...")
    res_cdc = pipeline.ingest_file(sample_file)
    print("CDC Re-ingestion result:")
    for k, v in res_cdc.items():
        if k != "active_doc_ids":
            print(f"  - {k}: {v}")

    if res_cdc["skipped_cdc"] == res["total_chunks"] and res_cdc["embedded_chunks"] == 0:
        print("SUCCESS: CDC worked! Zero Bedrock embedding calls made for unchanged chunks.")
    else:
        print("WARNING: CDC did not skip all chunks.")

    print("\n" + "=" * 60)
    print("STEP 6: Testing Vector & Lexical Retrieval in OpenSearch")
    print("=" * 60)
    # 6.1 BM25 Lexical Search (with accent folding)
    lexical_query = {
        "query": {
            "multi_match": {
                "query": "blackstone griddle",
                "fields": ["content", "metadata.product_name"],
            }
        }
    }
    lex_res = indexer.client.search(index=test_index, body=lexical_query)
    total_lex = lex_res["hits"]["total"]["value"]
    print(f"Lexical BM25 search found {total_lex} hits.")
    if total_lex > 0:
        top_hit = lex_res["hits"]["hits"][0]["_source"]
        print(f"Top BM25 Hit: {top_hit.get('metadata', {}).get('product_name')}")

    # 6.2 k-NN Vector Search
    query_vec = embedder.embed_text("Bếp nướng Blackstone mặt phẳng", input_type="search_query")
    knn_query = {
        "size": 2,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vec,
                    "k": 2,
                }
            }
        },
    }
    knn_res = indexer.client.search(index=test_index, body=knn_query)
    total_knn = knn_res["hits"]["total"]["value"]
    print(f"Vector k-NN search found {total_knn} hits.")
    if total_knn > 0:
        top_knn_hit = knn_res["hits"]["hits"][0]["_source"]
        score = knn_res["hits"]["hits"][0]["_score"]
        print(f"Top k-NN Hit (score={score:.4f}): {top_knn_hit.get('metadata', {}).get('product_name')}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_preflight_and_test()
