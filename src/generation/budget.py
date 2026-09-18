"""Context budgeting and token management for grounded LLM generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class BudgetReport:
    """Telemetry report of context trimming and token budgeting."""

    total_chunks_in: int
    chunks_retained: int
    chunks_dropped: int
    estimated_tokens: int
    is_truncated: bool = False


class ContextBudgetManager:
    """Manages input context length to fit within target Bedrock token limits."""

    def __init__(
        self,
        max_context_tokens: int = 4000,
        chars_per_token: float = 3.2,
        parent_max_chars: int = 2000,
    ) -> None:
        """
        Args:
            max_context_tokens: Token ceiling for retrieved context injected into prompt.
            chars_per_token: Average ratio of characters to tokens (3.0-3.5 for Vietnamese/English).
            parent_max_chars: Upper cap for parent chunk content.
        """
        self.max_context_tokens = max_context_tokens
        self.chars_per_token = chars_per_token
        self.parent_max_chars = parent_max_chars

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count from raw character length."""
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def budget_chunks(
        self, chunks: List[RetrievedChunk]
    ) -> Tuple[List[RetrievedChunk], BudgetReport]:
        """
        Trims and prunes chunks to strictly fit within the token budget.
        Prioritizes top-ranked chunks (assuming chunks are pre-sorted by rank/score).
        """
        if not chunks:
            return [], BudgetReport(
                total_chunks_in=0,
                chunks_retained=0,
                chunks_dropped=0,
                estimated_tokens=0,
                is_truncated=False,
            )

        retained: List[RetrievedChunk] = []
        accumulated_tokens = 0
        dropped = 0

        for chunk in chunks:
            content = chunk.content or ""
            # Apply safety cap on parent expanded text if applicable
            if chunk.is_parent_expanded and len(content) > self.parent_max_chars:
                content = content[: self.parent_max_chars]

            chunk_tokens = self.estimate_tokens(content)

            if accumulated_tokens + chunk_tokens <= self.max_context_tokens:
                retained.append(chunk)
                accumulated_tokens += chunk_tokens
            else:
                # If even the very first chunk exceeds max budget, keep a truncated version
                if not retained:
                    allowed_chars = int(self.max_context_tokens * self.chars_per_token)
                    truncated_content = content[:allowed_chars]
                    # Create a shallow copy with truncated content
                    chunk.content = truncated_content
                    retained.append(chunk)
                    accumulated_tokens = self.max_context_tokens
                dropped += 1

        is_truncated = dropped > 0
        if is_truncated:
            logger.warning(
                "Context budget exceeded: retained %d chunks, dropped %d chunks (estimated tokens: %d)",
                len(retained),
                dropped,
                accumulated_tokens,
            )

        report = BudgetReport(
            total_chunks_in=len(chunks),
            chunks_retained=len(retained),
            chunks_dropped=dropped,
            estimated_tokens=accumulated_tokens,
            is_truncated=is_truncated,
        )
        return retained, report
