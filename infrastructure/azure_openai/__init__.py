"""
Azure OpenAI Integration - Consolidated Client
All Azure OpenAI functionality consolidated in this module
"""

# Main unified client implementation
from .openai_client import UnifiedAzureOpenAIClient

# Maintain backwards compatibility with old class names
AzureOpenAIKnowledgeExtractor = UnifiedAzureOpenAIClient
AzureOpenAITextProcessor = UnifiedAzureOpenAIClient
AzureOpenAICompletionService = UnifiedAzureOpenAIClient
AzureOpenAIExtractionClient = UnifiedAzureOpenAIClient
OptimizedLLMExtractor = UnifiedAzureOpenAIClient

__all__ = [
    "UnifiedAzureOpenAIClient",
    "AzureOpenAIKnowledgeExtractor",
    "AzureOpenAITextProcessor",
    "AzureOpenAICompletionService",
    "AzureOpenAIExtractionClient",
    "OptimizedLLMExtractor",
]
