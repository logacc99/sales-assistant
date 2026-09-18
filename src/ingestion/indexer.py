"""OpenSearch vector indexing, CDC point lookups, and index lifecycle management."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from opensearchpy import OpenSearch, helpers

from src.config import get_config
from src.ingestion.models import EcomChunk

logger = logging.getLogger(__name__)


class BaseVectorIndexer(ABC):
    """Manages index lifecycle and document indexing via OpenSearch."""

    @abstractmethod
    def ensure_index_exists(self, index_name: str, recreate: bool = False) -> bool:
        """Creates the dual k-NN / BM25 index with appropriate mappings if it does not exist."""
        pass

    @abstractmethod
    def get_existing_hashes(self, index_name: str, doc_ids: list[str]) -> dict[str, str]:
        """Fetches existing {doc_id: content_hash} mappings from OpenSearch for CDC diffing."""
        pass

    @abstractmethod
    def index_chunks(
        self, index_name: str, chunks: list[EcomChunk]
    ) -> tuple[int, list[dict[str, Any]]]:
        """Bulk-indexes chunks via helpers.bulk. Returns (success_count, failed_items)."""
        pass

    @abstractmethod
    def purge_stale_documents(
        self, index_name: str, active_doc_ids: list[str], source_prefix: Optional[str] = None
    ) -> int:
        """Deletes documents in index matching source_prefix whose doc_ids are absent from active_doc_ids."""
        pass

    @abstractmethod
    def as_langchain_vectorstore(self, index_name: str) -> Any:
        """Returns a configured langchain_community.vectorstores.OpenSearchVectorSearch instance."""
        pass


class OpenSearchVectorIndexer(BaseVectorIndexer):
    """OpenSearch vector indexer supporting k-NN, Vietnamese lexical search, and CDC."""

    def __init__(
        self,
        client: Optional[OpenSearch] = None,
        dimension: Optional[int] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_aws_auth: Optional[bool] = None,
    ) -> None:
        cfg = get_config()
        self.dimension = dimension or cfg.bedrock_dimension
        self.host = host or cfg.opensearch_host
        self.port = port or cfg.opensearch_port
        self.use_aws_auth = use_aws_auth if use_aws_auth is not None else cfg.opensearch_use_aws_auth

        if client is not None:
            self.client = client
        else:
            self.client = self._create_opensearch_client()

    def _create_opensearch_client(self) -> OpenSearch:
        """Builds OpenSearch client using either AWS SigV4 or Basic/no-auth."""
        from urllib.parse import urlparse

        cfg = get_config()
        clean_host = self.host
        port = self.port
        use_ssl = "https" in self.host or self.port == 443

        if "://" in clean_host:
            parsed = urlparse(clean_host)
            clean_host = parsed.hostname or clean_host
            if parsed.port:
                port = parsed.port
            use_ssl = parsed.scheme == "https"
        elif ":" in clean_host:
            parts = clean_host.split(":")
            clean_host = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                port = int(parts[1])

        http_auth = None
        connection_class = None

        if self.use_aws_auth:
            try:
                import boto3
                session_kwargs: dict[str, Any] = {"region_name": cfg.aws_region}
                if cfg.aws_access_key_id and cfg.aws_secret_access_key:
                    session_kwargs["aws_access_key_id"] = cfg.aws_access_key_id
                    session_kwargs["aws_secret_access_key"] = cfg.aws_secret_access_key
                    if cfg.aws_session_token:
                        session_kwargs["aws_session_token"] = cfg.aws_session_token

                session = boto3.Session(**session_kwargs)
                credentials = session.get_credentials()

                if credentials:
                    try:
                        from opensearchpy import AWSV4SignerAuth, RequestsHttpConnection

                        http_auth = AWSV4SignerAuth(credentials, cfg.aws_region, "es")
                        connection_class = RequestsHttpConnection
                    except ImportError:
                        from requests_aws4auth import AWS4Auth
                        from opensearchpy import RequestsHttpConnection

                        frozen_creds = credentials.get_frozen_credentials()
                        http_auth = AWS4Auth(
                            frozen_creds.access_key,
                            frozen_creds.secret_key,
                            cfg.aws_region,
                            "es",
                            session_token=frozen_creds.token,
                        )
                        connection_class = RequestsHttpConnection
            except Exception as e:
                logger.warning(f"Could not initialize AWS SigV4 auth: {e}. Falling back to basic auth.")

        if http_auth is None and cfg.opensearch_username:
            http_auth = (cfg.opensearch_username, cfg.opensearch_password or "")

        client_kwargs: dict[str, Any] = {
            "hosts": [{"host": clean_host, "port": port}],
            "http_auth": http_auth,
            "use_ssl": use_ssl,
            "verify_certs": False,
            "ssl_show_warn": False,
        }
        if connection_class is not None:
            client_kwargs["connection_class"] = connection_class

        return OpenSearch(**client_kwargs)

    def get_index_mapping(self) -> dict[str, Any]:
        """Generates OpenSearch index configuration payload matching SPEC-0002."""
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                },
                "analysis": {
                    "analyzer": {
                        "vietnamese_ascii_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"],
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "content_hash": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "fields": {
                            "folded": {
                                "type": "text",
                                "analyzer": "vietnamese_ascii_analyzer",
                            }
                        },
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source_url": {"type": "keyword"},
                            "source_file": {"type": "keyword"},
                            "source_type": {"type": "keyword"},
                            "page_title": {"type": "text"},
                            "crawl_timestamp": {"type": "date"},
                            "product_id": {"type": "keyword"},
                            "product_name": {
                                "type": "text",
                                "fields": {
                                    "folded": {
                                        "type": "text",
                                        "analyzer": "vietnamese_ascii_analyzer",
                                    }
                                },
                            },
                            "price": {"type": "double"},
                            "currency": {"type": "keyword"},
                            "stock_status": {"type": "keyword"},
                            "policy_type": {"type": "keyword"},
                            "parent_section_content": {"type": "text", "index": False},
                            "promo_code": {"type": "keyword"},
                        },
                    },
                }
            },
        }

    def ensure_index_exists(self, index_name: str, recreate: bool = False) -> bool:
        """Creates index if it does not already exist, or recreates if recreate=True or mapping is invalid."""
        exists = self.client.indices.exists(index=index_name)

        if exists and not recreate:
            try:
                mapping = self.client.indices.get_mapping(index=index_name)
                if isinstance(mapping, dict):
                    props = (
                        mapping.get(index_name, {})
                        .get("mappings", {})
                        .get("properties", {})
                    )
                    emb_prop = props.get("embedding")
                    if isinstance(emb_prop, dict):
                        emb_type = emb_prop.get("type")
                        if emb_type and emb_type != "knn_vector":
                            logger.warning(
                                f"Index '{index_name}' exists but field 'embedding' is '{emb_type}' "
                                f"instead of 'knn_vector'. Recreating index with correct mappings..."
                            )
                            recreate = True
            except Exception as e:
                logger.warning(f"Could not verify existing index mapping for '{index_name}': {e}")

        if recreate and exists:
            logger.info(f"Deleting outdated index '{index_name}'...")
            self.client.indices.delete(index=index_name)
            exists = False

        if not exists:
            payload = self.get_index_mapping()
            self.client.indices.create(index=index_name, body=payload)
            logger.info(f"Created OpenSearch index '{index_name}' with dual k-NN and BM25 mapping.")
            return True

        return False

    def get_existing_hashes(self, index_name: str, doc_ids: list[str]) -> dict[str, str]:
        """Batch fetches existing {doc_id: content_hash} using OpenSearch _mget in batches of up to 500."""
        if not doc_ids:
            return {}

        try:
            if not self.client.indices.exists(index=index_name):
                return {}
        except Exception:
            return {}

        results: dict[str, str] = {}
        batch_size = 500

        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i : i + batch_size]
            docs_query = [{"_id": did, "_source": ["content_hash", "doc_id"]} for did in batch]
            try:
                response = self.client.mget(index=index_name, body={"docs": docs_query})
                for doc in response.get("docs", []):
                    if doc.get("found"):
                        source = doc.get("_source", {})
                        cid = doc.get("_id")
                        content_hash = source.get("content_hash")
                        if cid and content_hash:
                            results[cid] = content_hash
            except Exception as e:
                logger.warning(f"Error executing batch _mget on index {index_name}: {e}")

        return results

    def index_chunks(
        self, index_name: str, chunks: list[EcomChunk]
    ) -> tuple[int, list[dict[str, Any]]]:
        """Bulk-indexes chunks via helpers.bulk with raise_on_error=False."""
        if not chunks:
            return 0, []

        self.ensure_index_exists(index_name)

        actions = []
        for chunk in chunks:
            doc_body: dict[str, Any] = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
                "category": chunk.metadata.category,
                "content": chunk.content,
                "embedding": chunk.embedding,
                "metadata": {
                    "source_url": chunk.metadata.source_url,
                    "source_file": chunk.metadata.source_file,
                    "source_type": chunk.metadata.source_type,
                    "page_title": chunk.metadata.page_title,
                    "crawl_timestamp": (
                        chunk.metadata.crawl_timestamp.isoformat()
                        if chunk.metadata.crawl_timestamp
                        else None
                    ),
                    "product_id": (
                        chunk.metadata.product.product_id if chunk.metadata.product else None
                    ),
                    "product_name": (
                        chunk.metadata.product.name if chunk.metadata.product else None
                    ),
                    "price": chunk.metadata.product.price if chunk.metadata.product else None,
                    "currency": (
                        chunk.metadata.product.currency if chunk.metadata.product else None
                    ),
                    "stock_status": (
                        chunk.metadata.product.stock_status if chunk.metadata.product else None
                    ),
                    "policy_type": (
                        chunk.metadata.policy.policy_type if chunk.metadata.policy else None
                    ),
                    "parent_section_content": (
                        chunk.metadata.policy.parent_section_content
                        if chunk.metadata.policy
                        else None
                    ),
                    "promo_code": (
                        chunk.metadata.promotion.promo_code if chunk.metadata.promotion else None
                    ),
                },
            }

            actions.append(
                {
                    "_index": index_name,
                    "_id": chunk.id,
                    "_source": doc_body,
                }
            )

        success_count, errors = helpers.bulk(
            self.client,
            actions,
            raise_on_error=False,
            refresh=True,
        )

        return success_count, errors

    def purge_stale_documents(
        self, index_name: str, active_doc_ids: list[str], source_prefix: Optional[str] = None
    ) -> int:
        """Deletes indexed documents under source_prefix whose IDs are absent from active_doc_ids."""
        try:
            if not self.client.indices.exists(index=index_name):
                return 0
        except Exception:
            return 0

        must_filters: list[dict[str, Any]] = []
        if source_prefix:
            must_filters.append({"prefix": {"metadata.source_url": source_prefix}})

        must_not_filters = [{"terms": {"_id": active_doc_ids}}] if active_doc_ids else []

        query = {
            "query": {
                "bool": {
                    "must": must_filters,
                    "must_not": must_not_filters,
                }
            }
        }

        response = self.client.delete_by_query(
            index=index_name,
            body=query,
            refresh=True,
            wait_for_completion=True,
        )

        deleted = response.get("deleted", 0)
        logger.info(f"Purged {deleted} stale documents from index '{index_name}'.")
        return deleted

    def as_langchain_vectorstore(self, index_name: str, embedding_function: Optional[Any] = None) -> Any:
        """Returns a configured langchain_community.vectorstores.OpenSearchVectorSearch instance."""
        from langchain_community.vectorstores import OpenSearchVectorSearch

        return OpenSearchVectorSearch(
            opensearch_url=f"http://{self.host}:{self.port}",
            index_name=index_name,
            embedding_function=embedding_function,
            http_auth=self.client.transport.hosts[0].get("http_auth") if hasattr(self.client, "transport") else None,
            use_ssl="https" in self.host or self.port == 443,
            verify_certs=False,
        )
