"""Re-ranking interfaces, pluggable RerankerFactory, and provider implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
from typing import Any, Dict, List, Optional, Type

from src.config import IngestionConfig, get_config
from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for chunk re-ranking."""

    mode_name: str = "rerank"

    @abstractmethod
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Re-scores candidate chunks given the search query and returns top_k hits."""
        pass

    def initialize(self) -> None:
        """Optional hook to initialize weights or pre-warm connections on startup."""
        pass


class NoOpReranker(BaseReranker):
    """Pass-through reranker preserving incoming rank order."""

    mode_name: str = "noop_rerank"

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        selected = chunks[:top_k]
        for rank, chunk in enumerate(selected, start=1):
            chunk.rerank_rank = rank
            chunk.rerank_score = chunk.score
        return selected


class BedrockCohereReranker(BaseReranker):
    """AWS Bedrock Cohere Rerank client (cohere.rerank-v3-5:0)."""

    mode_name: str = "cohere_rerank"

    def __init__(
        self,
        model_id: Optional[str] = None,
        bedrock_client: Optional[Any] = None,
        region_name: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        self.model_id = model_id or cfg.bedrock_rerank_model_id
        self.region_name = region_name or cfg.bedrock_rerank_region

        if bedrock_client is not None:
            self.client = bedrock_client
        else:
            self.client = self._create_bedrock_client()

    def _create_bedrock_client(self) -> Any:
        """Initializes boto3 Bedrock Runtime client."""
        import boto3
        cfg = get_config()
        session_kwargs: dict[str, Any] = {"region_name": self.region_name}
        if cfg.aws_access_key_id and cfg.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = cfg.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = cfg.aws_secret_access_key
            if cfg.aws_session_token:
                session_kwargs["aws_session_token"] = cfg.aws_session_token

        session = boto3.Session(**session_kwargs)
        from botocore.config import Config as BotoConfig
        boto_config = BotoConfig(
            retries={"max_attempts": 1, "mode": "standard"},
            connect_timeout=3,
            read_timeout=5,
        )
        return session.client("bedrock-runtime", region_name=self.region_name, config=boto_config)

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Submits candidate chunks to Bedrock Cohere Rerank and re-orders hits."""
        if not chunks or top_k <= 0:
            return []

        # Prepare document texts for Cohere Rerank
        documents = [c.content for c in chunks]
        top_n = min(top_k, len(chunks))

        request_body = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "api_version": 2,
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
        except Exception as exc:
            logger.error(f"Bedrock Cohere reranking failed on model '{self.model_id}': {exc}")
            raise

        results = response_body.get("results", [])
        reranked_chunks: List[RetrievedChunk] = []

        for rank, res in enumerate(results, start=1):
            idx = res.get("index")
            rel_score = float(res.get("relevance_score", 0.0))
            if idx is not None and 0 <= idx < len(chunks):
                chunk = chunks[idx]
                chunk.rerank_score = rel_score
                chunk.rerank_rank = rank
                chunk.score = rel_score  # Update final score to reranker relevance score
                reranked_chunks.append(chunk)

        # Append any unranked chunks if top_n requested more than results returned
        if len(reranked_chunks) < top_n:
            seen_cids = {c.chunk_id for c in reranked_chunks}
            for chunk in chunks:
                if chunk.chunk_id not in seen_cids and len(reranked_chunks) < top_n:
                    reranked_chunks.append(chunk)

        return reranked_chunks


class LocalBGEReranker(BaseReranker):
    """In-process BAAI/bge-reranker-m3 (and compatible) cross-encoder."""

    mode_name: str = "local_bge_rerank"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        model_instance: Optional[Any] = None,
    ) -> None:
        cfg = get_config()
        self.model_id = model_id or getattr(cfg, "local_rerank_model_id", "BAAI/bge-reranker-m3")
        self.device = device if device is not None else getattr(cfg, "local_rerank_device", "")
        self.batch_size = batch_size if batch_size is not None else getattr(cfg, "local_rerank_batch_size", 16)
        self._model = model_instance

    def _resolve_device(self) -> str:
        """Resolves target compute device."""
        if self.device:
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_model(self) -> Any:
        """Loads cross-encoder model using sentence-transformers or FlagEmbedding."""
        resolved_dev = self._resolve_device()

        # 1. Try sentence_transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading local cross-encoder '{self.model_id}' on device '{resolved_dev}'...")
            return CrossEncoder(self.model_id, device=resolved_dev)
        except ImportError:
            pass
        except Exception as exc:
            logger.error(f"Failed loading CrossEncoder for '{self.model_id}': {exc}")
            raise

        # 2. Try FlagEmbedding FlagReranker
        try:
            from FlagEmbedding import FlagReranker
            use_fp16 = resolved_dev in ("cuda", "mps")
            logger.info(f"Loading FlagReranker '{self.model_id}' (use_fp16={use_fp16})...")
            return FlagReranker(self.model_id, use_fp16=use_fp16)
        except ImportError:
            pass
        except Exception as exc:
            logger.error(f"Failed loading FlagReranker for '{self.model_id}': {exc}")
            raise

        raise ImportError(
            "Neither 'sentence-transformers' nor 'FlagEmbedding' is installed. "
            "Please install sentence-transformers: pip install sentence-transformers"
        )

    def initialize(self) -> None:
        """Initializes model weights and runs warmup inference."""
        if self._model is None:
            self._model = self._load_model()

        # Warmup pass
        try:
            warmup_pairs = [["warmup query", "warmup document text"]]
            if hasattr(self._model, "predict"):
                self._model.predict(warmup_pairs, show_progress_bar=False)
            elif hasattr(self._model, "compute_score"):
                self._model.compute_score(warmup_pairs)
            logger.info(f"Local reranker '{self.model_id}' pre-warmed successfully.")
        except Exception as exc:
            logger.warning(f"Local reranker warmup encountered non-fatal issue: {exc}")

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Submits candidate chunks to local cross-encoder and re-orders hits."""
        if not chunks or top_k <= 0:
            return []

        if self._model is None:
            self._model = self._load_model()

        pairs = [[query, c.content] for c in chunks]

        try:
            if hasattr(self._model, "predict"):
                raw_scores = self._model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            elif hasattr(self._model, "compute_score"):
                raw_scores = self._model.compute_score(pairs, batch_size=self.batch_size)
            elif callable(self._model):
                raw_scores = self._model(pairs)
            else:
                raise TypeError(f"Incompatible model instance: {type(self._model)}")
        except Exception as exc:
            logger.error(f"Local reranker failed scoring query against {len(chunks)} chunks: {exc}")
            raise

        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]

        # Associate each chunk with its predicted relevance score
        scored_chunks = []
        for chunk, score in zip(chunks, raw_scores):
            scored_chunks.append((chunk, float(score)))

        # Sort descending by relevance score
        scored_chunks.sort(key=lambda item: item[1], reverse=True)

        top_n = min(top_k, len(scored_chunks))
        reranked_chunks: List[RetrievedChunk] = []

        for rank, (chunk, score) in enumerate(scored_chunks[:top_n], start=1):
            chunk.rerank_score = score
            chunk.rerank_rank = rank
            chunk.score = score
            reranked_chunks.append(chunk)

        return reranked_chunks


class RerankerFactory:
    """Factory and registry for pluggable reranker providers."""

    _registry: Dict[str, Type[BaseReranker]] = {
        "local": LocalBGEReranker,
        "bge": LocalBGEReranker,
        "bedrock": BedrockCohereReranker,
        "bedrock_cohere": BedrockCohereReranker,
        "cohere": BedrockCohereReranker,
        "noop": NoOpReranker,
    }

    @classmethod
    def register_reranker(cls, name: str, reranker_cls: Type[BaseReranker]) -> None:
        """
        Dynamically registers a new reranker provider class into the factory.

        Args:
            name: Method key (case-insensitive, e.g. "jina", "voyage", "llm").
            reranker_cls: Subclass of BaseReranker.
        """
        norm_name = name.strip().lower()
        cls._registry[norm_name] = reranker_cls
        logger.info("Registered reranker provider '%s' -> %s", norm_name, reranker_cls.__name__)

    @classmethod
    def get_registered_rerankers(cls) -> Dict[str, Type[BaseReranker]]:
        """Returns a copy of registered provider mappings."""
        return dict(cls._registry)

    @classmethod
    def create(
        cls,
        method: Optional[str] = None,
        app_config: Optional[IngestionConfig] = None,
        **kwargs: Any,
    ) -> BaseReranker:
        """
        Instantiates the requested or configured reranker provider.

        Args:
            method: Explicit method key ("local", "bedrock", "noop", or custom).
                    If None, reads from app_config.rerank_method (default: "local").
            app_config: IngestionConfig instance.
            **kwargs: Extra kwargs passed directly to provider constructor (e.g. model_instance).

        Returns:
            An instance of BaseReranker.

        Raises:
            ValueError: If the requested method is not registered.
        """
        config = app_config or get_config()
        selected_method = (
            method or getattr(config, "rerank_method", "local") or "local"
        ).strip().lower()

        reranker_cls = cls._registry.get(selected_method)
        if reranker_cls is None:
            available = sorted(list(cls._registry.keys()))
            raise ValueError(
                f"Unknown rerank method: '{selected_method}'. Registered providers: {available}"
            )

        if reranker_cls is NoOpReranker:
            return NoOpReranker()

        return reranker_cls(**kwargs)
