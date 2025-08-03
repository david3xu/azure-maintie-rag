"""
Azure OpenAI Completion Service - Compatibility Layer
This is a compatibility layer that aliases to the UnifiedAzureOpenAIClient
for backward compatibility with existing GNN modules.
"""

from .openai_client import UnifiedAzureOpenAIClient

# Backward compatibility alias
AzureOpenAIService = UnifiedAzureOpenAIClient

__all__ = ["AzureOpenAIService"]
