"""OpenSearch Query DSL Builder for BM25 lexical, Lucene k-NN dense vector, and _msearch payloads."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from src.retrieval.models import FilterCriteria


class OpenSearchQueryBuilder:
    """Constructs optimized OpenSearch query DSL payloads supporting Vietnamese text analysis and k-NN."""

    DEFAULT_BM25_FIELDS = [
        "content^2.0",
        "content.folded^1.5",
        "metadata.product_name^3.0",
        "metadata.product_name.folded^2.5",
    ]

    @staticmethod
    def build_filter_clauses(filters: Optional[FilterCriteria]) -> List[Dict[str, Any]]:
        """Converts FilterCriteria into a list of OpenSearch bool filter clauses."""
        if not filters:
            return []

        clauses: List[Dict[str, Any]] = []

        if filters.category:
            clauses.append({"term": {"category": filters.category}})

        if filters.stock_status:
            clauses.append({"term": {"metadata.stock_status": filters.stock_status}})

        if filters.policy_type:
            clauses.append({"term": {"metadata.policy_type": filters.policy_type}})

        if filters.promo_code:
            clauses.append({"term": {"metadata.promo_code": filters.promo_code}})

        if filters.source_prefix:
            clauses.append({"prefix": {"metadata.source_url": filters.source_prefix}})

        price_range: Dict[str, Any] = {}
        if filters.min_price is not None:
            price_range["gte"] = filters.min_price
        if filters.max_price is not None:
            price_range["lte"] = filters.max_price
        if price_range:
            clauses.append({"range": {"metadata.price": price_range}})

        return clauses

    @classmethod
    def build_bm25_query(
        cls,
        query: str,
        size: int = 15,
        filters: Optional[FilterCriteria] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Constructs BM25 multi-match query with Vietnamese accent-folding fields and filters."""
        search_fields = list(fields) if fields is not None else cls.DEFAULT_BM25_FIELDS
        filter_clauses = cls.build_filter_clauses(filters)

        query_body: Dict[str, Any] = {
            "size": size,
            "_source": True,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": search_fields,
                                "operator": "or",
                                "minimum_should_match": "2<75%",
                            }
                        }
                    ],
                }
            },
        }

        if filter_clauses:
            query_body["query"]["bool"]["filter"] = filter_clauses

        return query_body

    @classmethod
    def build_knn_query(
        cls,
        query_vector: List[float],
        size: int = 15,
        filters: Optional[FilterCriteria] = None,
        vector_field: str = "embedding",
    ) -> Dict[str, Any]:
        """Constructs Lucene HNSW k-NN dense vector query with filters."""
        filter_clauses = cls.build_filter_clauses(filters)

        knn_clause = {
            "knn": {
                vector_field: {
                    "vector": query_vector,
                    "k": size,
                }
            }
        }

        query_body: Dict[str, Any] = {
            "size": size,
            "_source": True,
            "query": {
                "bool": {
                    "must": [knn_clause],
                }
            },
        }

        if filter_clauses:
            query_body["query"]["bool"]["filter"] = filter_clauses

        return query_body

    @classmethod
    def build_msearch_body(
        cls,
        index_name: str,
        bm25_query: Dict[str, Any],
        knn_query: Dict[str, Any],
    ) -> str:
        """Serializes BM25 and k-NN queries into OpenSearch _msearch NDJSON payload.
        
        Format:
          {"index": index_name}\n
          {bm25_query_json}\n
          {"index": index_name}\n
          {knn_query_json}\n
        """
        lines: List[str] = [
            json.dumps({"index": index_name}),
            json.dumps(bm25_query),
            json.dumps({"index": index_name}),
            json.dumps(knn_query),
        ]
        return "\n".join(lines) + "\n"
