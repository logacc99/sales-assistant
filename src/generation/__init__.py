"""Bedrock LLM response generation subsystem for Sales Assistant."""

from src.generation.bedrock_client import BaseBedrockClient, BedrockConverseClient
from src.generation.budget import BudgetReport, ContextBudgetManager
from src.generation.citation_extractor import CitationExtractor
from src.generation.generator import GroundedResponseGenerator
from src.generation.models import (
    CartItem,
    Citation,
    CitationType,
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    ShopperContext,
    StreamChunk,
    TokenUsage,
)
from src.generation.prompts import PromptBuilder
from src.generation.service import GenerationService

__all__ = [
    "BaseBedrockClient",
    "BedrockConverseClient",
    "BudgetReport",
    "ContextBudgetManager",
    "CitationExtractor",
    "GroundedResponseGenerator",
    "GenerationService",
    "CartItem",
    "Citation",
    "CitationType",
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResponse",
    "ShopperContext",
    "StreamChunk",
    "TokenUsage",
    "PromptBuilder",
]
