"""External service integrations for Universal RAG system.

This package contains integrations with external services like
Azure OpenAI, vector databases, graph databases, and monitoring.
"""

from .azure_openai_wrapper import AzureOpenAIClient
# azure_manager removed to avoid circular imports - services are now independent

__all__ = [
    'AzureOpenAIClient'
]