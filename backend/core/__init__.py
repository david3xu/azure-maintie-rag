"""Universal RAG Core - Azure Services Architecture"""

# Azure OpenAI Services
from .azure_openai import (
    AzureOpenAICompletionService,
    AzureOpenAITextProcessor,
    AzureOpenAIKnowledgeExtractor,
    AzureOpenAIExtractionClient
)

# Azure Search Services
from .azure_search import (
    AzureSearchVectorService,
    AzureSearchQueryAnalyzer
)

# Azure ML Services
from .azure_ml import (
    AzureMLGNNProcessor,
    AzureMLClassificationService
)

# Azure Orchestration Services
from .orchestration import (
    AzureRAGOrchestrationService,
    AzureRAGEnhancedPipeline
)

__all__ = [
    # Azure service components
    'AzureOpenAICompletionService', 'AzureOpenAITextProcessor',
    'AzureOpenAIKnowledgeExtractor', 'AzureOpenAIExtractionClient',
    'AzureSearchVectorService', 'AzureSearchQueryAnalyzer',
    'AzureMLGNNProcessor', 'AzureMLClassificationService',
    'AzureRAGOrchestrationService', 'AzureRAGEnhancedPipeline'
]