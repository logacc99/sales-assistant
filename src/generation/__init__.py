"""LLM response generation subsystem for Sales Assistant supporting Bedrock Runtime and Mantle."""

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
from src.generation.providers import (
    BaseBedrockClient,
    BaseLLMClient,
    BedrockConverseClient,
    BedrockMantleClient,
    BedrockRuntimeClient,
    LLMClientFactory,
)
from src.generation.service import GenerationService

__all__ = [
    "BaseLLMClient",
    "BaseBedrockClient",
    "LLMClientFactory",
    "BedrockRuntimeClient",
    "BedrockConverseClient",
    "BedrockMantleClient",
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
