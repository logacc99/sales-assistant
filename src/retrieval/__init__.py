"""Retrieval subsystem package: hybrid search, RRF fusion, Cohere reranking, and query building."""

from src.retrieval.fusion import LinearCombinationFusion, ReciprocalRankFusion
from src.retrieval.models import (
    FilterCriteria,
    FusionAlgorithm,
    RerankerType,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
    SearchType,
)
from src.retrieval.query_builder import OpenSearchQueryBuilder
from src.retrieval.reranker import BaseReranker, BedrockCohereReranker, NoOpReranker
from src.retrieval.retriever import BaseRetriever, OpenSearchHybridRetriever
from src.retrieval.service import RetrievalService

__all__ = [
    "BaseRetriever",
    "OpenSearchHybridRetriever",
    "RetrievalService",
    "OpenSearchQueryBuilder",
    "ReciprocalRankFusion",
    "LinearCombinationFusion",
    "BaseReranker",
    "BedrockCohereReranker",
    "NoOpReranker",
    "RetrievalQuery",
    "RetrievedChunk",
    "RetrievalResult",
    "FilterCriteria",
    "SearchType",
    "FusionAlgorithm",
    "RerankerType",
]
