# SPEC-0000: Sales Assistant — Overall System Architecture Specification

**Spec ID**: `SPEC-0000`  
**Status**: `Approved`  
**Scope**: Overall System Architecture, Subsystem Boundaries, and Governance  
**Date**: 2026-09-18  

---

## 1. Executive Summary & Vision

The **Sales Assistant** is an enterprise-grade Retrieval-Augmented Generation (RAG) conversational agent engineered for e-commerce stores. The system enables natural-language customer inquiries regarding products, verified pricing, active promotions, and store policies (shipping, returns, warranties) while guaranteeing zero hallucinations through strict grounding and real-time inventory verification.

### 1.1 Core Business Goals
- **Verified Product Discovery**: Provide instant answers on product specs, live pricing, and stock status.
- **Deterministic Promotion Guidance**: Present valid coupon discounts, calculate maximum savings without stacking, and enforce minimum spend thresholds.
- **Grounded Policy Clarity**: Answer shipping, warranty, and return inquiries with direct links to official policies.
- **Merchant Integrity**: Politely refuse out-of-stock purchases or invalid coupons, redirecting shoppers to support when appropriate.

### 1.2 Non-Goals
- Direct in-chat credit card payment processing (assistant provides checkout CTAs).
- Arbitrary non-e-commerce general chit-chat (assistant redirects to store domain topics).
- Hardcoding domain data in prompt context (all knowledge must be grounded in OpenSearch or live hooks).

---

## 2. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Data Sources & Ingestion [Ingestion & Indexing Pipeline: SPEC-0002]
        Web[E-commerce Crawler - Crawl4AI] --> IngestPipeline[Ingestion Engine]
        Policies[Store Policy Markdown] --> IngestPipeline
        IngestPipeline --> Chunker[Semantic Chunker]
        Chunker --> BedrockEmbed[AWS Bedrock Embeddings]
        BedrockEmbed --> OpenSearch[(Amazon OpenSearch Service Dual Store)]
    end

    subgraph Runtime & Server Gateway [Unified ASGI Runtime: main.py]
        CLI["CLI Entrypoint: python main.py"] --> Uvicorn["Uvicorn ASGI Process Manager"]
        Uvicorn --> API["FastAPI Gateway: /api/v1/*, /health, /docs"]
        API --> IngestPipeline
        API --> SearchVerification[Search Test & Verification]
    end

    subgraph Assistant Orchestration [E-Commerce Assistant: SPEC-0001]
        UserQuery[Shopper Query + Cart Context] --> AssistantAgent[LangGraph StateGraph Engine]
        AssistantAgent --> Classifier[Intent & Category Classifier]
        Classifier --> HybridRetrieval[Hybrid Search: BM25 + k-NN 1024-d]
        OpenSearch -.-> HybridRetrieval
        HybridRetrieval --> LiveHook[LiveInventoryService Hook]
        LiveHook --> DomainRules[Promo & Policy Rules Engine]
        DomainRules --> GeneratorLLM[AWS Bedrock LLM Grounded Generator]
        GeneratorLLM --> AssistantPayload[Structured AssistantResponse + UI Cards]
    end
```

---

## 3. Subsystem Specification Hierarchy

The system strictly adheres to **Spec-Driven Development (SDD)**. Low-level technical contracts, Pydantic schemas, embedding dimensions, index mappings, and API endpoints are formally owned by their respective subsystem specifications:

| Spec ID | Subsystem | Specification File | Responsibilities |
| :--- | :--- | :--- | :--- |
| **`SPEC-0000`** | **Overall System & Server Runtime** | [`specs/00-system-spec.md`](specs/00-system-spec.md) | High-level topology, runtime CLI entrypoint (`main.py`), pre-flight diagnostics, subsystem boundaries, and cross-cutting SLAs. |
| **`SPEC-0001`** | **Assistant Orchestration** | [`specs/01-ecommerce-assistant-spec.md`](specs/01-ecommerce-assistant-spec.md) | LangGraph StateGraph, shopper context, promotion evaluation rules, live stock verification hooks, and UI citations. |
| **`SPEC-0002`** | **Ingestion & Indexing** | [`specs/02-ingestion-indexing-pipeline-spec.md`](specs/02-ingestion-indexing-pipeline-spec.md) | Crawl4AI crawling, markdown chunking, Bedrock Cohere v3 embeddings, OpenSearch k-NN lucene schema, Change Data Detection (CDC), and REST API. |
| **`SPEC-0003`** | **Retrieval & Re-ranking** | [`specs/03-hybrid-retrieval-reranking-spec.md`](specs/03-hybrid-retrieval-reranking-spec.md) *(Approved)* | Hybrid fusion (BM25 + Dense k-NN via OpenSearch `_msearch` + client-side RRF), Bedrock Cohere re-ranking, and parent-child policy expansion. |
| **`SPEC-0004`** | **Bedrock LLM Generation** | [`specs/04-llm-response-generation-spec.md`](specs/04-llm-response-generation-spec.md) *(Draft)* | Grounded prompt synthesis using AWS Bedrock Converse API, context token budgeting, structured UI citation cards, and polite refusal policies. |

---

## 4. Cross-Cutting & Non-Functional Requirements

### 4.1 Cloud Infrastructure
- **Vector Storage**: Amazon OpenSearch Service domain equipped with native Lucene HNSW k-NN indexing and Vietnamese ASCII text analysis.
- **Foundation Models**: AWS Bedrock hosting `cohere.embed-multilingual-v3.0` (1024-d vectors) for dense representations and Claude / Nova for grounded synthesis.
- **Authentication**: AWS SigV4 by default with environment-driven fallback to Basic Auth for local testing.

### 4.2 Language & Tone Resilience
- **Vietnamese Diacritic Resiliency**: Hybrid search analyzers must handle shoppers typing with or without tone marks (e.g. `bep nuong` matching `bếp nướng`).
- **Brand & SKU Precision**: Exact lexical keywords (BM25) ensure high fidelity for technical part numbers, SKUs, and brand names.

### 4.3 Performance & Latency Budgets
- **Ingestion Run**: Change Data Detection (CDC) slashes recurring ingestion API calls by $\ge 90\%$.
- **Retrieval Latency**: OpenSearch hybrid point query $\le 100\text{ ms}$.
- **End-to-End Chat SLA**: Complete assistant response generation within $\le 2.0\text{ s}$.

---

## 5. Architectural Governance & SDD Workflow

1. **Spec as Single Source of Truth**: No feature code may be introduced in `src/` without referencing an approved spec document.
2. **Contract-First Implementation**: Data schemas, index mappings, and public interfaces must be agreed upon in the dedicated subsystem spec prior to implementation.
3. **Spec Updates on Pivot**: Any changes in requirements, data contracts, or architecture must be immediately reflected in the corresponding `specs/*.md` document.

---

## 6. Runtime Serving & Unified API Entrypoint Specification

### 6.1 Architectural Objective & Core Principles
The root entrypoint (`main.py`) serves as the single source of operational execution for the entire Sales Assistant system. It unifies server startup, pre-flight diagnostics, environment validation, and ASGI process supervision.

1. **Zero-Friction Developer Experience**: Executing `python main.py` must boot the entire REST gateway with sensible defaults without requiring lengthy command-line switches.
2. **Deterministic Pre-flight Verification**: Before opening network sockets, the entrypoint validates critical configuration (AWS region, credentials, OpenSearch connectivity) to prevent runtime failures deep inside request lifecycles.
3. **Operational Transparency (Banner DX)**: Displays an informative startup banner detailing API version, bound host/port, interactive docs links (`/docs`, `/redoc`), active vector indices, and registered route groups.
4. **Graceful Termination**: Captures `SIGINT` (Ctrl+C) and `SIGTERM` signals, completing in-flight transactions and closing client pools cleanly.

### 6.2 CLI Specification & Parameters

The entrypoint must provide an expressive command-line interface:

| Argument / Flag | Env Variable Fallback | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `APP_HOST` / `HOST` | `127.0.0.1` | Network interface address to bind server. |
| `--port` | `APP_PORT` / `PORT` | `8000` | TCP port to listen on. |
| `--reload` / `--no-reload` | `APP_RELOAD` | `True` in local dev | Enable/disable auto-reloading on code changes. |
| `--workers` | `APP_WORKERS` | `1` | Number of worker processes (enforced `1` if `--reload` is active). |
| `--log-level` | `APP_LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warning`, `error`). |
| `--check` | — | `False` | Run pre-flight health diagnostics only and exit without launching HTTP listener. |

### 6.3 Pre-flight Diagnostics Protocol

When launched (or when invoked with `--check`), the startup protocol executes the following validation pipeline:

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Developer / Systemd / Docker
    participant Main as main.py (Entrypoint)
    participant Cfg as IngestionConfig (.env)
    participant OS as OpenSearch Cluster
    participant Bedrock as AWS Bedrock Runtime

    Operator->>Main: python main.py [--check]
    Main->>Cfg: Load configuration & environment
    alt Configuration Missing or Malformed
        Main-->>Operator: Exit code 1 with actionable error message
    end

    Main->>OS: Ping cluster / Check index metadata
    alt OpenSearch Unreachable
        Main-->>Operator: Display connectivity warning or fail fast
    end

    Main->>Bedrock: Validate AWS Region & Credentials
    alt AWS Bedrock Misconfigured
        Main-->>Operator: Display AWS credentials warning
    end

    opt If --check was passed
        Main-->>Operator: Print Diagnostics Report & Exit (Code 0 or 1)
    end

    Main->>Operator: Print ASCII Startup Banner & Route Summary
    Main->>Main: uvicorn.run("src.api.main:app", host, port, reload, workers)
```

### 6.4 Terminal Banner & Route Discovery Contract

Upon successful initialization, `main.py` prints a clean diagnostic overview:

```text
================================================================================
  🛍️  SALES ASSISTANT — API GATEWAY & RAG RUNTIME
================================================================================
  Status:           Online (Ready for requests)
  API Version:      v0.1.0
  Listening Address: http://127.0.0.1:8000
  Swagger UI:       http://127.0.0.1:8000/docs
  ReDoc:            http://127.0.0.1:8000/redoc
  OpenSearch Host:  localhost:9200 (Index: sales-assistant-catalog)
  Bedrock Model:    cohere.embed-multilingual-v3.0 (1024-d)
  Active Routes:
    - [GET]  /health
    - [POST] /api/v1/crawler/crawl
    - [POST] /api/v1/ingestion/run
    - [POST] /api/v1/ingestion/index-raw
    - [POST] /api/v1/ingestion/search
================================================================================
```

### 6.5 Verification & Acceptance Criteria
- **AC-1 (Help Flag)**: `python main.py --help` outputs standard argument flags and exits with code `0`.
- **AC-2 (Dry-Run Diagnostics)**: `python main.py --check` evaluates environment, OpenSearch client, and Bedrock embedder initialization, reporting exact status.
- **AC-3 (Live Boot)**: Executing `python main.py` starts the server on specified port, logs the startup banner, and successfully handles `GET /health` responding `{"success": true}` with HTTP 200.
- **AC-4 (Signal Handling)**: Sending `SIGINT` (Ctrl+C) gracefully halts Uvicorn without traceback leaks.

---

## 7. Master Environment Variable Reference & System Configuration

As a core Spec-Driven Development (SDD) rule, all runtime parameters across every subsystem must be explicitly specified. The following master reference table anchors all variables across `.env`, `.env.example`, and `src/config.py`:

| Subsystem | Variable Name | Type | Default Value | Defining Spec | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Server Runtime** | `APP_HOST` | `str` | `127.0.0.1` | `SPEC-0000` (Sec 6.2) | Bind network interface address for Uvicorn ASGI server. |
| **Server Runtime** | `APP_PORT` | `int` | `8000` | `SPEC-0000` (Sec 6.2) | TCP port for Uvicorn ASGI server. |
| **Server Runtime** | `APP_RELOAD` | `bool` | `true` | `SPEC-0000` (Sec 6.2) | Enable hot-reload on file modification (development mode). |
| **Server Runtime** | `APP_WORKERS` | `int` | `1` | `SPEC-0000` (Sec 6.2) | Worker process count (enforced to 1 when `APP_RELOAD` is true). |
| **Server Runtime** | `APP_LOG_LEVEL` | `str` | `info` | `SPEC-0000` (Sec 6.2) | Logging verbosity (`debug`, `info`, `warning`, `error`). |
| **AWS & Auth** | `AWS_REGION` | `str` | `us-east-1` | `SPEC-0002` (Sec 2.4) | Primary AWS region for Bedrock Converse and OpenSearch SigV4 auth. |
| **AWS & Auth** | `AWS_ACCESS_KEY_ID` | `str` | `None` | `SPEC-0002` (Sec 2.4) | AWS Access Key ID (optional if using IAM Role or SSO). |
| **AWS & Auth** | `AWS_SECRET_ACCESS_KEY`| `str` | `None` | `SPEC-0002` (Sec 2.4) | AWS Secret Access Key. |
| **AWS & Auth** | `AWS_SESSION_TOKEN` | `str` | `None` | `SPEC-0002` (Sec 2.4) | AWS Session Token for temporary IAM STS credentials. |
| **OpenSearch** | `OPENSEARCH_HOST` | `str` | `localhost` | `SPEC-0002` (Sec 2.4) | OpenSearch Serverless collection HTTPS URL or cluster host. |
| **OpenSearch** | `OPENSEARCH_PORT` | `int` | `443` | `SPEC-0002` (Sec 2.4) | Port (`443` for AOSS Serverless HTTPS; `9200` for local dev). |
| **OpenSearch** | `OPENSEARCH_INDEX_NAME`| `str` | `sales-assistant-catalog` | `SPEC-0002` (Sec 2.4) | Target OpenSearch k-NN index name. |
| **OpenSearch** | `OPENSEARCH_IS_SERVERLESS` | `bool` | Auto (`true` for `.aoss.`) | `SPEC-0002` (Sec 2.4) | Enables Serverless behaviors (signing service `aoss`, omits cluster APIs). |
| **OpenSearch** | `OPENSEARCH_SERVICE_NAME` | `str` | Auto (`aoss` / `es`) | `SPEC-0002` (Sec 2.4) | AWS SigV4 signing service (`aoss` for Serverless, `es` for provisioned). |
| **OpenSearch** | `OPENSEARCH_COLLECTION_TYPE` | `str` | `VECTORSEARCH` | `SPEC-0002` (Sec 2.4) | OpenSearch Serverless collection type (must be `VECTORSEARCH` for k-NN). |
| **OpenSearch** | `OPENSEARCH_USE_AWS_AUTH` | `bool` | `true` | `SPEC-0002` (Sec 2.4) | Enables AWS SigV4 request signing with IAM credentials. |
| **OpenSearch** | `OPENSEARCH_USERNAME` | `str` | `admin` | `SPEC-0002` (Sec 2.4) | Basic auth username (used only when `OPENSEARCH_USE_AWS_AUTH=false`). |
| **OpenSearch** | `OPENSEARCH_PASSWORD` | `str` | `admin` | `SPEC-0002` (Sec 2.4) | Basic auth password (used only when `OPENSEARCH_USE_AWS_AUTH=false`). |
| **Ingestion** | `BEDROCK_EMBEDDING_MODEL_ID` | `str` | `cohere.embed-multilingual-v3.0` | `SPEC-0002` (Sec 2.4) | AWS Bedrock embedding model ID. |
| **Ingestion** | `BEDROCK_EMBEDDING_DIMENSION` | `int` | `1024` | `SPEC-0002` (Sec 2.4) | Embedding vector dimension (must match index mapping). |
| **Ingestion** | `BEDROCK_MAX_WORKERS` | `int` | `5` | `SPEC-0002` (Sec 2.4) | Thread pool size for parallel document embedding. |
| **Retrieval** | `BEDROCK_RERANK_MODEL_ID` | `str` | `cohere.rerank-v3-5:0` | `SPEC-0003` (Sec 2.4) | AWS Bedrock model ID for Cohere Rerank v3.5. |
| **Retrieval** | `BEDROCK_RERANK_REGION` | `str` | `ap-northeast-1` (fallback: `AWS_REGION`) | `SPEC-0003` (Sec 2.4) | Region for Bedrock Rerank API (e.g. `ap-northeast-1`, `us-west-2`). |
| **Generation** | `LLM_METHOD` | `str` | `runtime` | `SPEC-0004` (Sec 2.3) | Provider routing: `"runtime"` (Converse API) or `"mantle"` (OpenAI gateway). |
| **Generation** | `BEDROCK_GENERATION_MODEL_ID` | `str` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | `SPEC-0004` (Sec 2.3) | Default model for Bedrock Converse API (`LLM_METHOD=runtime`). |
| **Generation** | `BEDROCK_GENERATION_TEMPERATURE` | `float` | `0.1` | `SPEC-0004` (Sec 2.3) | Sampling temperature for generation. |
| **Generation** | `BEDROCK_GENERATION_MAX_TOKENS` | `int` | `1500` | `SPEC-0004` (Sec 2.3) | Maximum token output limit. |
| **Generation** | `BEDROCK_GENERATION_TIMEOUT_SECONDS` | `int` | `30` | `SPEC-0004` (Sec 2.3) | HTTP read/socket timeout for generation requests. |
| **Generation (Mantle)** | `BEDROCK_MANTLE_API_KEY` | `str` | `""` | `SPEC-0004` (Sec 2.3) | Bearer token for Bedrock Mantle OpenAI-compatible gateway. |
| **Generation (Mantle)** | `BEDROCK_MANTLE_BASE_URL` | `str` | `""` | `SPEC-0004` (Sec 2.3) | Mantle endpoint (e.g. `https://bedrock-mantle.ap-southeast-2.api.aws/v1`). |
| **Generation (Mantle)** | `BEDROCK_MANTLE_PROJECT_ID` | `str` | `default` | `SPEC-0004` (Sec 2.3) | Project ID sent as query parameter (`?project_id=...`). |
| **Generation (Mantle)** | `BEDROCK_MANTLE_MODEL_ID` | `str` | `openai.gpt-oss-120b` | `SPEC-0004` (Sec 2.3) | Default model identifier for Mantle gateway calls. |

