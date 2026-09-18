# Hybrid Retrieval & Re-ranking Specification

**Spec ID**: `SPEC-0003`  
**Status**: `Approved`  
**Author**: Antigravity & User  
**Date**: 2026-09-19 (Aligned with OpenSearch Serverless)  
**Parent Spec**: [`specs/00-system-spec.md`](file:///home/ngthuan/projects/sales-assistant/specs/00-system-spec.md), [`specs/01-ecommerce-assistant-spec.md`](file:///home/ngthuan/projects/sales-assistant/specs/01-ecommerce-assistant-spec.md)  
**Related Specs**: [`specs/02-ingestion-indexing-pipeline-spec.md`](file:///home/ngthuan/projects/sales-assistant/specs/02-ingestion-indexing-pipeline-spec.md)

---

## 1. Problem Statement & Motivation

### 1.1 Context
In the Sales Assistant architecture, customer inquiries range from precise technical requests (e.g., exact SKU codes like `"BBQ-GAS-01"`, brand names like `"Weber"`, or specific dimensions) to fuzzy semantic inquiries (e.g., `"bếp dùng ngoài trời cho gia đình 4 người"` or `"chính sách đổi trả hàng bị lỗi"`). 

Furthermore, Vietnamese e-commerce search presents unique challenges:
1. **Diacritic & Tone Variations**: Shoppers frequently omit tone marks (e.g., typing `"bep nuong ga ngoai troi"` instead of `"bếp nướng gas ngoài trời"`).
2. **Compound Term Semantics**: Vietnamese words often span multiple tokens (e.g., `"nồi chiên không dầu"`), where pure BM25 can match disjointed terms, while pure dense vector retrieval may miss exact part numbers or brand variants.
3. **Multi-Domain Granularity**: Products require structured metadata filtering (price, stock, category), promotions require threshold and validity checking, and policies require parent-child context expansion so the LLM receives the full policy section without losing chunk retrieval precision.

To achieve $\ge 85\%$ Hit Rate @ 3 and high-precision grounded answers, we require a dedicated, high-performance **Hybrid Retrieval & Re-ranking Engine** backed by **Amazon OpenSearch Serverless (AOSS collection: `VECTORSEARCH`)** and AWS Bedrock.

### 1.2 User Story
As an e-commerce conversational assistant (and backend engineer testing via REST), I want a unified retrieval module that executes hybrid search (BM25 with Vietnamese accent-folding + Lucene HNSW k-NN vectors) against an Amazon OpenSearch Serverless collection using `_msearch`, fuses scores via client-side Reciprocal Rank Fusion (RRF), re-ranks candidate pools with Bedrock Cohere Rerank (`cohere.rerank-v3-5:0`), applies metadata filters (category, stock, price), expands parent-child policy contexts on final hits (capped at 2,000 chars), and provides full graceful degradation telemetry, so that the assistant receives high-precision, grounded context with zero hallucinations and zero cluster maintenance overhead.

### 1.3 Non-Goals / Out of Scope
- **Final LLM Response Generation**: Governed by `SPEC-0001` (LangGraph assistant workflow and prompt synthesis).
- **Index Lifecycle & Chunk Ingestion**: Governed by `SPEC-0002` (crawling, chunking, Bedrock bulk embedding, and CDC indexing).
- **Live Inventory Mutation**: Real-time stock reservation or checkout cart mutations (retrieval inspects catalog stock status; real-time transactional locking is handled by downstream order systems).

---

## 2. Technical Architecture & Data Contracts

### 2.1 Data Models & Schemas (`src/retrieval/models.py`)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class SearchType(str, Enum):
    """Retrieval execution strategy."""
    HYBRID = "hybrid"
    VECTOR = "vector"
    LEXICAL = "lexical"


class FusionAlgorithm(str, Enum):
    """Rank fusion algorithm used to merge BM25 and dense vector results."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR_COMBINATION = "linear"  # Normalized weighted sum: alpha * dense + (1 - alpha) * bm25


class RerankerType(str, Enum):
    """Reranker implementation."""
    BEDROCK_COHERE = "bedrock_cohere"
    NOOP = "noop"


@dataclass
class FilterCriteria:
    """Structured search filters for OpenSearch queries."""
    category: Optional[Literal["product", "promotion", "policy"]] = None
    stock_status: Optional[Literal["in_stock", "out_of_stock", "backorder"]] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    policy_type: Optional[Literal["shipping", "returns", "warranty", "privacy", "terms"]] = None
    promo_code: Optional[str] = None
    source_prefix: Optional[str] = None


@dataclass
class RetrievalQuery:
    """Input query specification for hybrid retrieval."""
    query: str
    search_type: SearchType = SearchType.HYBRID
    top_k: int = 5
    filters: Optional[FilterCriteria] = None
    fusion_algorithm: FusionAlgorithm = FusionAlgorithm.RRF
    rrf_k: int = 60
    hybrid_alpha: float = 0.5  # Weight of dense vector in linear fusion (0.0 = pure BM25, 1.0 = pure vector)
    score_threshold: float = 0.0  # Minimum similarity/fusion score cutoff
    expand_parent_context: bool = True  # If True, replace content with parent_section_content on final top-K hits
    parent_max_chars: int = 2000  # Safety cap for injected parent content
    rerank: bool = True  # Always-on Bedrock Cohere Rerank by default
    rerank_model_id: str = "cohere.rerank-v3-5:0"
    reranker_type: RerankerType = RerankerType.BEDROCK_COHERE

    @property
    def candidate_pool_size(self) -> int:
        """Dynamic candidate pool size fetched from OpenSearch for reranking: min(max(top_k * 3, 15), 30)."""
        return min(max(self.top_k * 3, 15), 30)


@dataclass
class RetrievedChunk:
    """Individual chunk hit retrieved, scored, and re-ranked."""
    chunk_id: str
    doc_id: str
    category: str
    content: str
    score: float
    lexical_score: Optional[float] = None
    vector_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_content: Optional[str] = None
    is_parent_expanded: bool = False


@dataclass
class RetrievalResult:
    """Full result bundle from the retrieval engine with degradation telemetry."""
    query: str
    search_type: SearchType
    total_hits: int
    chunks: List[RetrievedChunk]
    latency_ms: float
    retrieval_mode_used: str
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    degradation_reason: Optional[str] = None
```

---

### 2.2 Component Interactions & Retrieval Pipeline

```mermaid
flowchart TD
    UserQuery[Shopper Query: e.g. 'bep nuong gas ngoai troi'] --> Embedder[AWS Bedrock Embedder: Cohere Multilingual v3 - search_query]
    UserQuery --> QueryBuilder[OpenSearch Query Builder]
    
    subgraph Single HTTPS Round-Trip: OpenSearch Serverless _msearch (service: aoss)
        QueryBuilder -->|BM25 Header + Body| MSearchPayload[OpenSearch _msearch NDJSON Payload]
        Embedder -->|Dense 1024-d Vector Header + Body| MSearchPayload
        MSearchPayload --> OpenSearchServerless[(Amazon OpenSearch Serverless: VECTORSEARCH)]
        OpenSearchServerless --> MSearchResponse[OpenSearch _msearch Response]
    end
    
    MSearchResponse --> SplitHits[Separate BM25 & k-NN Candidate Lists]
    SplitHits --> ClientFusion[Client-Side Fusion Engine: RRF k=60 in Python]
    ClientFusion --> DedupPool[Deduplicated Candidate Pool: min max top_k*3, 15, 30]
    
    DedupPool --> RerankCheck{Rerank Enabled & Healthy?}
    RerankCheck -->|Yes| BedrockRerank[AWS Bedrock Cohere Rerank: cohere.rerank-v3-5:0]
    RerankCheck -->|No / Fallback| RRFOrder[Fallback to RRF Rank Order + Mark Degraded]
    
    BedrockRerank --> TopKSlice[Slice Top-K Winning Chunks]
    RRFOrder --> TopKSlice
    
    TopKSlice --> ParentExpander[Parent-Child Policy Expander: max 2,000 chars]
    ParentExpander --> FinalContext[Retrieved Context Bundle -> LangGraph Assistant]
```

---

### 2.3 Retrieval Algorithms & Core Architectural Decisions

#### 2.3.1 Bedrock Embedding for Search Queries
- In `SPEC-0002`, documents were embedded with `input_type="search_document"`.
- At query time, `BedrockEmbedder.embed_text(query, input_type="search_query")` **must** be used to ensure asymmetrical retrieval alignment in Cohere Multilingual v3 (1024-d).

#### 2.3.2 Candidate Retrieval via OpenSearch Serverless `_msearch`
To eliminate unnecessary network latency while preserving clean independent score sets on Amazon OpenSearch Serverless:
1. The retriever generates two query bodies:
   - **Header 1 / Body 1**: BM25 lexical query matching `content^2.0`, `content.folded^1.5`, `metadata.product_name^3.0`, `metadata.product_name.folded^2.5`, with active category/stock/price filters.
   - **Header 2 / Body 2**: k-NN vector query against `embedding` vector (1024-d) with the exact same filters applied in `filter` clause.
2. Both queries request `size = candidate_pool_size`.
3. Dispatched in a single HTTPS network request using `opensearch.msearch()` signed with AWS SigV4 (`service="aoss"`). OpenSearch Serverless natively executes multi-search across `VECTORSEARCH` collection indices without requiring cluster plugin installations.

#### 2.3.3 Client-Side Reciprocal Rank Fusion (RRF)
Fusing BM25 candidates and vector candidates in Python (`src/retrieval/fusion.py`):
$$RRF(d) = \sum_{m \in \{bm25, vector\}} \frac{1}{k + r_m(d)}$$
Where:
- $k = 60$ (constant smoothing parameter).
- $r_m(d)$ is the 1-based rank of chunk $d$ in result set $m$. If document $d$ did not appear in the top candidate pool for modality $m$, its score component for $m$ is $0$.

#### 2.3.4 Always-On Re-ranking via Bedrock Cohere Rerank
- Model: `cohere.rerank-v3-5:0` invoked via AWS Bedrock Runtime.
- **Candidate Pool**: The top $N = \min(\max(top\_k \times 3, 15), 30)$ deduplicated candidates from RRF are submitted to Bedrock Cohere Rerank.
- Reranker receives the query and list of chunk `content` strings.
- Re-orders the candidate pool by `relevance_score` descending.

#### 2.3.5 Post-Ranking Parent-Child Policy Expansion
- **Why Post-Ranking?**: Scoring on the granular chunk ensures Cohere Rerank evaluates the exact clause answering the user's question, preserving token efficiency and high relevance scoring.
- **Expansion Rule**: Once the final top-$K$ chunks are selected, any chunk with `category == "policy"` that contains `metadata.parent_section_content` has its `content` replaced by the parent section content, truncated to a maximum of `parent_max_chars` (default 2,000 characters). `is_parent_expanded` is set to `True`.

---

## 3. Interfaces & Architecture Contracts

### 3.1 Module Hierarchy (`src/retrieval/`)

```
src/retrieval/
├── __init__.py                # Package exports (OpenSearchHybridRetriever, RetrievalQuery, etc.)
├── models.py                  # Pydantic & dataclass schemas for query, filters, results
├── query_builder.py           # OpenSearch DSL generator (BM25, k-NN, _msearch payloads, filters)
├── fusion.py                  # Client-side RRF and Min-Max score normalization algorithms
├── reranker.py                # BaseReranker, BedrockCohereReranker, NoOpReranker
├── retriever.py               # BaseRetriever and OpenSearchHybridRetriever core service
└── service.py                 # High-level RetrievalService orchestrating cache, telemetry, and metrics
```

### 3.2 Public Python Method Signatures

```python
class BaseRetriever(ABC):
    """Abstract interface for Sales Assistant knowledge retrieval."""
    
    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Executes retrieval according to the provided query specification."""
        pass
    
    @abstractmethod
    async def aretrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Asynchronous retrieval execution."""
        pass


class OpenSearchHybridRetriever(BaseRetriever):
    """Production hybrid retriever combining OpenSearch _msearch + Bedrock k-NN + Cohere Rerank."""
    
    def __init__(
        self,
        opensearch_client: Optional[OpenSearch] = None,
        embedder: Optional[BaseEmbedder] = None,
        reranker: Optional[BaseReranker] = None,
        index_name: Optional[str] = None,
        default_top_k: int = 5,
    ) -> None:
        ...

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Synchronous hybrid retrieval with graceful degradation."""
        ...

    async def aretrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Asynchronous hybrid retrieval."""
        ...

    def search_by_product_id(self, product_id: str) -> Optional[RetrievedChunk]:
        """Direct point lookup for deterministic product verification."""
        ...

    def search_promotions(self, min_spend: Optional[float] = None) -> List[RetrievedChunk]:
        """Retrieves currently active coupons matching spend criteria."""
        ...
```

### 3.3 REST API Specification (`src/api/v1/endpoints/retrieval.py`)

The retrieval engine provides dedicated REST verification endpoints to inspect retrieval quality, rank scoring, and re-ranking performance directly via OpenAPI Swagger UI (`/docs`).

#### 3.3.1 `POST /api/v1/retrieval/search`
- **Purpose**: Primary hybrid retrieval endpoint for assistant orchestration and manual verification.
- **Request Body**:
  ```json
  {
    "query": "bep nuong gas ngoai troi cao cap",
    "search_type": "hybrid",
    "top_k": 5,
    "filters": {
      "category": "product",
      "stock_status": "in_stock",
      "min_price": 2000000.0,
      "max_price": 15000000.0
    },
    "fusion_algorithm": "rrf",
    "rrf_k": 60,
    "score_threshold": 0.0,
    "expand_parent_context": true,
    "rerank": true
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "success": true,
    "message": "Retrieved 3 chunks in 88.4ms",
    "data": {
      "query": "bep nuong gas ngoai troi cao cap",
      "search_type": "hybrid",
      "total_hits": 3,
      "latency_ms": 88.4,
      "retrieval_mode_used": "hybrid_msearch_rrf_cohere_rerank",
      "degraded": false,
      "degradation_reason": null,
      "chunks": [
        {
          "chunk_id": "chunk_7f8a9b1c",
          "doc_id": "doc_bep_gas_01",
          "category": "product",
          "score": 0.962,
          "rerank_score": 0.962,
          "rerank_rank": 1,
          "rrf_score": 0.0328,
          "lexical_score": 14.82,
          "vector_score": 0.824,
          "lexical_rank": 1,
          "vector_rank": 2,
          "content": "Tên sản phẩm: Bếp nướng gas ngoài trời BBQ...",
          "metadata": {
            "product_name": "Bếp nướng gas ngoài trời BBQ 4 họng",
            "price": 8500000.0,
            "currency": "VND",
            "stock_status": "in_stock",
            "source_url": "https://bepbbq.com/san-pham/bep-nuong-gas-01"
          },
          "is_parent_expanded": false
        }
      ]
    }
  }
  ```

#### 3.3.2 `POST /api/v1/retrieval/explain`
- **Purpose**: Diagnostics endpoint providing transparent visibility into score calculations: raw BM25 score, raw vector cosine score, intermediate RRF reciprocal ranks, Cohere rerank scores, and degradation status.

#### 3.3.3 `GET /api/v1/retrieval/health`
- **Purpose**: Diagnostics check confirming OpenSearch Serverless connectivity, target collection index existence, mapping status, Bedrock embedding API availability, and Bedrock Cohere Rerank accessibility.
- **Serverless Behavior**: In OpenSearch Serverless, cluster-level APIs (e.g. `GET /`, `_cluster/health`) return 400/404 or are restricted by data access policies. The health check evaluates connectivity via `client.indices.exists(index=target_index)` and reports `is_serverless: true` and `collection_type: "VECTORSEARCH"`.
- **Response**: `200 OK`
  ```json
  {
    "success": true,
    "message": "Retrieval subsystem status: healthy",
    "data": {
      "status": "healthy",
      "opensearch_connected": true,
      "index_exists": true,
      "index_name": "sales-assistant-index",
      "document_count": 142,
      "bedrock_embedder_ready": true,
      "bedrock_reranker_ready": true,
      "is_serverless": true,
      "collection_type": "VECTORSEARCH"
    }
  }
  ```

---

## 4. Edge Cases, Failure Resilience & Graceful Degradation

1. **Tone-Mark Resiliency (Vietnamese Diacritics)**:
   - Queries like `"noi chien khong dau"` must match indexed documents containing `"Nồi chiên không dầu"`. The OpenSearch query builder targets both the raw `content` field and the `.folded` subfield generated with `vietnamese_ascii_analyzer`.
2. **SKU & Alphanumeric Model Codes**:
   - Exact SKU queries (e.g. `"BBQ-104-SS"`) often produce low semantic vector similarities. BM25 boosting (`metadata.product_name^3.0` and exact keyword matching) ensures that precise codes achieve Rank 1 in hybrid RRF.
3. **Empty / Whitespace / Punctuation Queries**:
   - Validation rejects empty or whitespace-only queries with HTTP 400 (`Query string cannot be empty`).
4. **Zero Relevant Hits / Sub-Threshold Results**:
   - If no candidates exceed `score_threshold`, return an empty list gracefully (`total_hits: 0, chunks: []`) with HTTP 200 rather than throwing an exception.
5. **Bedrock Embedding Outage or Throttling**:
   - If AWS Bedrock embedding fails, log an error, fall back to pure BM25 lexical search, set `degraded = True` and `degradation_reason = "Bedrock embedding unavailable, degraded to BM25"`. The customer query succeeds without crashing.
6. **Bedrock Cohere Rerank Outage or Throttling**:
   - If Cohere Rerank fails (e.g. AWS throttling or service outage), log a warning, fall back directly to the RRF rank order, set `degraded = True` and `degradation_reason = "Cohere rerank failed, degraded to RRF order"`.
7. **Parent-Child Missing Parent or Large Content**:
   - If `expand_parent_context=True` but `metadata.parent_section_content` is null or empty, the retriever retains the original chunk `content`. If present, content is capped at `parent_max_chars` (default 2,000 characters) to avoid LLM context overflow.
8. **OpenSearch Serverless Diagnostics & Root Ping Limitation**:
   - OpenSearch Serverless does not support cluster-level root ping (`HEAD /` or `GET /`) or `_cluster/health`. The retriever client and `/health` probe verify connectivity through collection index queries (`client.indices.exists(index=target_index)`), ensuring resilience against false-positive offline alerts.
9. **OpenSearch Serverless AWS SigV4 Scoping & Scaling**:
   - All retrieval operations dispatch signed HTTPS requests with `service="aoss"`. Query compute automatically scales via search OpenSearch Compute Units (OCUs) without requiring manual shard or replica re-allocation.

---

## 5. Acceptance Criteria & Evaluation Plan

### 5.1 Automated Acceptance Criteria
- [ ] **Unit Tests**:
  - `tests/test_opensearch_query_builder.py`: Validates generated OpenSearch JSON query DSL for `_msearch`, hybrid, vector, BM25, and metadata filters.
  - `tests/test_fusion.py`: Tests RRF ranking correctness, tie-breaking, and linear min-max normalization.
  - `tests/test_retrieval_models.py`: Validates Pydantic/dataclass serialization, deserialization, and filter constraints.
  - `tests/test_reranker.py`: Validates `BedrockCohereReranker` and `NoOpReranker` ordering and score propagation.
- [ ] **Integration Tests**:
  - `tests/test_opensearch_hybrid_retriever.py`: End-to-end tests against OpenSearch mock and live index.
  - `tests/test_retrieval_api.py`: FastAPI test client testing `/api/v1/retrieval/search`, `/explain`, and `/health` (including serverless indicators).
  - `tests/test_retrieval_fallback.py`: Verifies graceful fallback to BM25 if Bedrock query embedding raises an error, and fallback to RRF if Cohere rerank fails.

### 5.2 RAG Quality Metrics (`rag-eval` Skill)
Benchmarked via `.agent/skills/rag-eval/scripts/eval_retrieval.py` against `tests/fixtures/eval_dataset.json`:
- **Hit Rate @ 3**: $\ge 85\%$ (Target $\ge 90\%$ for product inquiries).
- **Mean Reciprocal Rank (MRR)**: $\ge 0.75$.
- **P95 Retrieval Latency**: $\le 150\text{ ms}$ for hybrid retrieval without reranking; $\le 400\text{ ms}$ with Bedrock Cohere re-ranking.
- **Tone-mark Equivalence**: Hit rate delta between accented and unaccented queries $\le 5\%$.

---

## 6. Architecture Decisions Log (Resolved via `/grill-me`)

| Decision Area | Selected Choice | Rationale |
| :--- | :--- | :--- |
| **OpenSearch Target** | Amazon OpenSearch Serverless (AOSS) Collection (`VECTORSEARCH`) | Eliminates provisioned cluster node/master sizing, auto-scales search OCUs, and natively supports `_msearch` and `knn_vector` without cluster plugin maintenance. |
| **Fusion Layer** | Client-side Python RRF (`src/retrieval/fusion.py`) | Maximum portability across OpenSearch versions, zero cluster plugin overhead, unit-testable without live clusters, and complete score transparency. |
| **OpenSearch Querying** | Multi-Search API (`_msearch`) | Dispatches independent BM25 and k-NN vector queries in a single HTTP network round-trip, halving connection overhead. |
| **Re-ranking Layer** | Always-On Bedrock Cohere Rerank (`cohere.rerank-v3-5:0`) | Delivers state-of-the-art multi-lingual semantic alignment and filtering of low-relevance candidates. |
| **Candidate Pool Size** | Dynamic: $\min(\max(top\_k \times 3, 15), 30)$ | Balances candidate recall with Bedrock Cohere API latency and token cost. |
| **Parent-Child Context Expansion** | Re-rank on granular chunk; expand parent context on final top-$K$ hits (capped at 2,000 chars) | Cohere Rerank evaluates the exact matching clause for optimal precision, while the assistant LLM receives the full policy section. |
| **Failure Resilience & Telemetry** | Full graceful degradation with `degraded: bool` telemetry | Bedrock embedding errors degrade to BM25; Cohere rerank errors degrade to RRF order. Assistant sessions never crash. |
