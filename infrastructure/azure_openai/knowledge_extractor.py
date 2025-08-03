"""
Azure OpenAI Knowledge Extractor - Compatibility Layer
This is a compatibility layer that aliases to the UnifiedAzureOpenAIClient
for backward compatibility with existing GNN modules.
"""

from .openai_client import UnifiedAzureOpenAIClient

# Backward compatibility alias
AzureOpenAIKnowledgeExtractor = UnifiedAzureOpenAIClient

__all__ = ["AzureOpenAIKnowledgeExtractor"]
