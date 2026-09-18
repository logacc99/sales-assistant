"""Unit tests for OpenSearchQueryBuilder DSL generation and _msearch formatting."""

import json
from src.retrieval.models import FilterCriteria
from src.retrieval.query_builder import OpenSearchQueryBuilder


def test_build_filter_clauses():
    filters = FilterCriteria(
        category="product",
        stock_status="in_stock",
        min_price=500000.0,
        max_price=2000000.0,
        policy_type="warranty",
        promo_code="SUMMER2026",
        source_prefix="https://bepbbq.com/san-pham",
    )
    clauses = OpenSearchQueryBuilder.build_filter_clauses(filters)
    assert len(clauses) == 6

    clause_map = {}
    for c in clauses:
        if "term" in c:
            clause_map.update(c["term"])
        elif "prefix" in c:
            clause_map.update(c["prefix"])
        elif "range" in c:
            clause_map.update(c["range"])

    assert clause_map["category"] == "product"
    assert clause_map["metadata.stock_status"] == "in_stock"
    assert clause_map["metadata.policy_type"] == "warranty"
    assert clause_map["metadata.promo_code"] == "SUMMER2026"
    assert clause_map["metadata.source_url"] == "https://bepbbq.com/san-pham"
    assert clause_map["metadata.price"] == {"gte": 500000.0, "lte": 2000000.0}


def test_build_bm25_query():
    filters = FilterCriteria(category="product")
    q = OpenSearchQueryBuilder.build_bm25_query(
        query="bep nuong than hoa",
        size=10,
        filters=filters,
    )
    assert q["size"] == 10
    assert q["_source"] is True
    must_clause = q["query"]["bool"]["must"][0]["multi_match"]
    assert must_clause["query"] == "bep nuong than hoa"
    assert "metadata.product_name^3.0" in must_clause["fields"]
    assert "content.folded^1.5" in must_clause["fields"]
    assert must_clause["operator"] == "or"
    assert must_clause["minimum_should_match"] == "2<75%"

    filter_clause = q["query"]["bool"]["filter"]
    assert len(filter_clause) == 1
    assert filter_clause[0] == {"term": {"category": "product"}}


def test_build_knn_query():
    dummy_vec = [0.1] * 1024
    filters = FilterCriteria(stock_status="in_stock")
    q = OpenSearchQueryBuilder.build_knn_query(
        query_vector=dummy_vec,
        size=15,
        filters=filters,
    )
    assert q["size"] == 15
    must_clause = q["query"]["bool"]["must"][0]["knn"]["embedding"]
    assert must_clause["k"] == 15
    assert len(must_clause["vector"]) == 1024

    filter_clause = q["query"]["bool"]["filter"]
    assert filter_clause[0] == {"term": {"metadata.stock_status": "in_stock"}}


def test_build_msearch_body():
    bm25 = {"size": 5, "query": {"match_all": {}}}
    knn = {"size": 5, "query": {"match_all": {}}}
    index_name = "sales-assistant-catalog"

    ndjson = OpenSearchQueryBuilder.build_msearch_body(index_name, bm25, knn)
    lines = ndjson.strip().split("\n")
    assert len(lines) == 4

    h1 = json.loads(lines[0])
    b1 = json.loads(lines[1])
    h2 = json.loads(lines[2])
    b2 = json.loads(lines[3])

    assert h1 == {"index": "sales-assistant-catalog"}
    assert b1 == bm25
    assert h2 == {"index": "sales-assistant-catalog"}
    assert b2 == knn
