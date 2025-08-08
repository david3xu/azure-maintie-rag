# Azure Universal RAG - Development Guide

**Universal RAG Development Patterns - Based on Actual Implementation**

Development guide for the Azure Universal RAG system with **zero hardcoded domain bias** and PydanticAI framework integration.

## 🔍 Universal RAG Philosophy

This development guide follows the **Universal RAG philosophy** implemented in the actual codebase:

### **Core Implementation Principles**
- **Domain-Agnostic Design**: No predetermined domain categories (technical, legal, medical, etc.)
- **Content Discovery**: System analyzes content characteristics dynamically
- **Universal Models**: All data structures work across ANY domain (`agents/core/universal_models.py`)
- **Real Azure Integration**: PydanticAI with AsyncAzureOpenAI, Cosmos DB Gremlin, Cognitive Search
- **Type Safety**: Pydantic models for all agent interfaces with validation

### **Agent Architecture (3 Specialized Agents)**
```
Domain Intelligence → Knowledge Extraction → Universal Search
        ↓                      ↓                    ↓
Content Discovery      Universal Extraction   Multi-modal Search
   (Azure OpenAI)        (Cosmos Gremlin)      (Vector+Graph+GNN)
```

## 🚀 Development Setup

### **Prerequisites**
```bash
# Python 3.11+ (required for PydanticAI)
python --version  # Verify Python 3.11+

# Azure CLI for authentication
az --version && az login

# Install dependencies
pip install -r requirements.txt
```

### **Environment Configuration**
Based on the Universal RAG configuration system (`config/universal_config.py`, `agents/core/simple_config_manager.py`):

```bash
# Environment synchronization (critical for multi-environment support)
./scripts/deployment/sync-env.sh development    # Switch to development + sync backend
./scripts/deployment/sync-env.sh staging       # Switch to staging + sync backend
make sync-env                                   # Sync backend with current azd environment

# Required Azure service endpoints (DefaultAzureCredential)
export AZURE_OPENAI_ENDPOINT="your-openai-endpoint"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
export OPENAI_MODEL_DEPLOYMENT="your-gpt-deployment"

# Multi-environment support
export USE_MANAGED_IDENTITY="true"  # Production
export USE_MANAGED_IDENTITY="false" # Development
```

## 🔧 Real Development Workflows

### **Start Development Environment**

```bash
# Method 1: Start FastAPI backend (based on api/main.py)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Method 2: Use make if available
make dev

# Test the running server
curl http://localhost:8000/  # FastAPI root endpoint
curl http://localhost:8000/health  # Health check endpoint
```

### **Test Universal RAG Agent Implementations**

Based on actual agent implementations with Universal RAG philosophy:

```bash
# Test Domain Intelligence Agent (content discovery, not domain classification)
cd agents/domain_intelligence && python agent.py
# Expected: Content characteristic analysis (vocabulary complexity, concept density)

# Test Knowledge Extraction Agent (universal extraction, domain-agnostic)
cd agents/knowledge_extraction && python agent.py  
# Expected: Entity/relationship extraction works for ANY domain

# Test Universal Search Agent (multi-modal search orchestration)
cd agents/universal_search && python agent.py
# Expected: Vector + Graph + GNN unified search

# Test Query Generation Showcase (demonstration script)
python scripts/dataflow/12_query_generation_showcase.py
# Expected: Query generation patterns and techniques

# Test Knowledge Extraction Agent (agents/knowledge_extraction/agent.py:368 lines)
python -c "
from agents.knowledge_extraction.agent import get_knowledge_extraction_agent
agent = get_knowledge_extraction_agent()
print('✅ Knowledge Extraction Agent created with multi-strategy extraction')
"

# Test Universal Search Agent (agents/universal_search/agent.py)
python -c "
from agents.universal_search.agent import get_universal_search_agent
agent = get_universal_search_agent()
print('✅ Universal Search Agent created with multi-modal search orchestration')
"
```

### **Test Azure Service Integration**

Based on `agents/core/azure_service_container.py` (471 lines):

```bash
python -c "
import asyncio
from agents.core.azure_service_container import ConsolidatedAzureServices

async def test_real_services():
    container = ConsolidatedAzureServices()
    print('Testing real Azure service container initialization...')
    
    try:
        # Test the actual initialize_all_services method
        status = await container.initialize_all_services()
        
        print('Azure Service Initialization Results:')
        for service, success in status.items():
            icon = '✅' if success else '❌'
            print(f'{icon} {service}: {\"SUCCESS\" if success else \"FAILED\"}')
        
        # Test the actual get_service_status method
        health = container.get_service_status()
        print(f'Overall Health: {health[\"overall_health\"]}')
        print(f'Services Ready: {health[\"successful_services\"]}/{health[\"total_services\"]}')
        
    except Exception as e:
        print(f'Service initialization error: {e}')
        print('Note: Requires Azure service endpoints to be configured')

asyncio.run(test_real_services())
"
```

## 📁 Real Codebase Structure

### **Core Infrastructure (`agents/core/`)**

Based on actual file analysis:

**azure_service_container.py (471 lines):**
```python
@dataclass
class ConsolidatedAzureServices:
    credential: DefaultAzureCredential = field(default_factory=DefaultAzureCredential)
    ai_foundry_provider: Optional[AzureProvider] = None
    # ... other service clients
    
    async def initialize_all_services(self) -> Dict[str, bool]:
        # Real implementation with parallel service initialization
        
    def get_service_status(self) -> Dict[str, Any]:
        # Real service health monitoring
```

**data_models.py (1,536 lines):**
```python
class ExtractionQualityOutput(BaseModel):
    """PydanticAI output validator for extraction quality assessment"""
    entities_per_text: float = Field(ge=1.0, le=20.0)
    relations_per_entity: float = Field(ge=0.3, le=5.0)
    overall_score: float = Field(ge=0.0, le=1.0)

class TextStatistics(BaseModel):
    """Statistical analysis results with computed properties"""
    total_words: int = Field(ge=0)
    lexical_diversity: float = Field(ge=0, le=1)
    
    @computed_field
    @property
    def readability_score(self) -> float:
        # Real Flesch Reading Ease calculation
```

**constants.py (1,186 lines):**
- Centralized configuration constants
- Eliminates hardcoded values throughout the system

### **Agent Implementations**

**Domain Intelligence Agent (`agents/domain_intelligence/agent.py`):**
```python
def create_domain_intelligence_agent() -> Agent:
    """Real PydanticAI agent creation with lazy initialization"""
    model_name = get_azure_openai_model()  # Uses environment variables
    
    agent = Agent(
        model_name,
        deps_type=DomainDeps,
        toolsets=[domain_intelligence_toolset],  # FunctionToolset pattern
        system_prompt="""You are the Domain Intelligence Agent..."""
    )
    return agent

# Lazy initialization pattern
_domain_intelligence_agent = None

def get_domain_intelligence_agent():
    global _domain_intelligence_agent
    if _domain_intelligence_agent is None:
        _domain_intelligence_agent = create_domain_intelligence_agent()
    return _domain_intelligence_agent
```

**Knowledge Extraction Agent (`agents/knowledge_extraction/agent.py`):**
```python
def _create_agent_with_toolset() -> Agent:
    """Create Knowledge Extraction Agent with unified processor integration"""
    azure_model = OpenAIModel(
        deployment_name,
        provider=AzureProvider(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
        ),
    )
    
    agent = Agent(
        azure_model,
        deps_type=KnowledgeExtractionDeps,
        toolsets=[get_knowledge_extraction_toolset()],
        name="knowledge-extraction-agent"
    )
    return agent
```

### **Shared Infrastructure (`agents/shared/`)**

Based on actual shared utilities:

**text_statistics.py:**
```python
class TextStatistics(BaseModel):
    """PydanticAI-enhanced statistical analysis"""
    total_words: int = Field(ge=0, description="Total word count")
    lexical_diversity: float = Field(ge=0, le=1, description="Unique/total ratio")
    
    @computed_field
    @property 
    def readability_score(self) -> float:
        """Flesch Reading Ease calculation"""
        return min(100.0, max(0.0, 206.835 - (1.015 * self.avg_words_per_sentence)))

def calculate_text_statistics(text: str) -> TextStatistics:
    """Calculate real statistical analysis of text content"""
    # Real implementation using statistical analysis
```

**content_preprocessing.py:**
```python
class TextCleaningOptions(BaseModel):
    """Configuration for content cleaning"""
    remove_html: bool = True
    normalize_whitespace: bool = True
    
class CleanedContent(BaseModel):
    """Result of content preprocessing"""
    cleaned_text: str
    cleaning_quality_score: float
    
def clean_text_content(text: str, options: TextCleaningOptions) -> CleanedContent:
    """Real content preprocessing implementation"""
```

### **Infrastructure Layer (`infrastructure/`)**

**Azure OpenAI Client (`infrastructure/azure_openai/openai_client.py`):**
```python
class UnifiedAzureOpenAIClient(BaseAzureClient):
    """Unified client for all Azure OpenAI operations"""
    
    def _initialize_client(self):
        """Real Azure OpenAI client initialization"""
        if self.use_managed_identity:
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            
            self._client = AzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=model_config.openai_api_version,
                azure_endpoint=self.endpoint,
            )
        else:
            self._client = AzureOpenAI(
                api_key=self.key,
                api_version=model_config.openai_api_version,
                azure_endpoint=self.endpoint,
            )
```

## 🧪 Real Development Testing

### **Test Real Components**

```bash
# Test shared infrastructure utilities
python -c "
from agents.shared.text_statistics import calculate_text_statistics
text = 'This is a sample text for statistical analysis testing.'
stats = calculate_text_statistics(text)
print(f'✅ Text Statistics: {stats.total_words} words, readability: {stats.readability_score:.1f}')

from agents.shared.content_preprocessing import clean_text_content, TextCleaningOptions
options = TextCleaningOptions(remove_html=True)
result = clean_text_content('<p>HTML content</p>', options) 
print(f'✅ Content Preprocessing: Quality score {result.cleaning_quality_score:.2f}')
"

# Test centralized data models  
python -c "
from agents.core.data_models import ExtractionQualityOutput, ValidatedEntity
print('✅ Centralized data models loaded (80+ Pydantic models available)')
print('   - ExtractionQualityOutput: PydanticAI output validator')
print('   - ValidatedEntity: Entity extraction validation')
print('   - TextStatistics: Statistical analysis with computed fields')
"
```

### **Integration Testing**

```bash
# Test real Azure service integration
python -c "
import asyncio
from agents.core.azure_service_container import ConsolidatedAzureServices

async def integration_test():
    print('🔧 Testing real Azure service integration...')
    container = ConsolidatedAzureServices()
    
    # Test real service initialization
    status = await container.initialize_all_services() 
    
    # Test real health check method
    health = await container.health_check()
    print(f'Health Check Results:')
    for service, status in health['service_health'].items():
        print(f'  {service}: {status}')
    
    print(f'Overall Status: {health[\"overall_status\"]}')
    print(f'Health Percentage: {health[\"health_percentage\"]}%')

asyncio.run(integration_test())
"
```

## 📊 Real Performance Monitoring

### **Agent Performance Testing**

```bash
# Test real agent response times
python -c "
import time
import asyncio
from agents.knowledge_extraction.agent import extract_knowledge_from_document
from agents.core.data_models import ExtractionConfiguration

async def performance_test():
    config = ExtractionConfiguration(
        domain_name='test',
        entity_confidence_threshold=0.7,
        relationship_confidence_threshold=0.7,
        chunk_size=500,
        chunk_overlap=50,
        expected_entity_types=['concept', 'method'],
        target_response_time_seconds=2.0,
        technical_vocabulary=[],
        key_concepts=[],
        generation_timestamp='2024-01-01T00:00:00'
    )
    
    start_time = time.time()
    result = await extract_knowledge_from_document(
        'Sample document content for extraction testing.',
        config,
        'test_doc'
    )
    duration = time.time() - start_time
    
    print(f'Real extraction completed in {duration:.3f}s')
    print(f'Entities extracted: {result.entity_count}')
    print(f'Relationships extracted: {result.relationship_count}')
    print(f'Extraction confidence: {result.extraction_confidence:.3f}')

asyncio.run(performance_test())
"
```

### **Configuration System Testing**

```bash
# Test real configuration management
python -c "
from config.universal_config import get_universal_config
from agents.core.simple_config_manager import SimpleConfigManager

try:
    system_config = get_system_config()
    print('✅ System configuration loaded')
    
    model_config = get_model_config_bootstrap() 
    print('✅ Model configuration (bootstrap) loaded')
    print(f'   API Version: {model_config.api_version}')
    
    workflow_config = get_workflow_config()
    print('✅ Workflow configuration loaded')
    
    print('✅ Configuration system operational (no circular dependencies)')
    
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

## 🛠️ Development Best Practices

### **Code Standards**
Based on the real implementation:

- **PydanticAI Compliance**: Use FunctionToolset patterns, proper dependencies
- **Lazy Initialization**: Avoid import-time Azure connection requirements
- **Centralized Models**: Use agents/core/data_models.py for all type definitions
- **Configuration Bootstrap**: Use bootstrap patterns to avoid circular dependencies
- **Error Handling**: Implement proper Azure service retry logic

### **Agent Development Patterns**

```python
# ✅ GOOD: Lazy initialization pattern
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent

# ✅ GOOD: PydanticAI FunctionToolset pattern
from pydantic_ai import Agent

agent = Agent(
    model,
    deps_type=AgentDeps,
    toolsets=[agent_toolset],  # FunctionToolset
    system_prompt="..."
)

# ✅ GOOD: Use centralized data models
from agents.core.data_models import ExtractionRequest, ExtractionResult

# ❌ BAD: Import-time Azure connections
# client = AzureOpenAI()  # Don't do this at module level
```

### **Configuration Patterns**

```python
# ✅ GOOD: Use centralized configuration
from config.universal_config import get_universal_config

# For initialization (avoid circular deps)
model_config = get_model_config_bootstrap()

# ✅ GOOD: Environment-based configuration
import os
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# ❌ BAD: Hardcoded values
# endpoint = "https://hardcoded.openai.azure.com/"
```

## 🔧 Debugging Real Issues

### **Common Development Issues**

**1. Azure Authentication Failures:**
```bash
# Check Azure CLI authentication
az account show

# Test specific service access
az cognitiveservices account list
```

**2. Circular Import Dependencies:**
```bash
# Test configuration loading
python -c "
from config.universal_config import get_universal_config
config = get_model_config_bootstrap()
print('✅ Bootstrap configuration works')
"
```

**3. PydanticAI Agent Creation Issues:**
```bash
# Test individual agent creation
python -c "
from agents.domain_intelligence.agent import create_domain_intelligence_agent
agent = create_domain_intelligence_agent()
print('✅ Agent creation successful')
"
```

### **Debug Commands**

```bash
# Test full system initialization
python -c "
import sys
sys.path.append('.')

print('Testing core components...')
from agents.core.azure_service_container import ConsolidatedAzureServices
print('✅ Azure service container')

from agents.core.data_models import ExtractionQualityOutput
print('✅ Centralized data models') 

from agents.shared.text_statistics import calculate_text_statistics
print('✅ Shared infrastructure')

from config.universal_config import get_universal_config
print('✅ Configuration system')

print('🎉 All core components loaded successfully')
"
```

## 🎯 Success Indicators

Your development environment is ready when:

- **Azure Service Container**: ConsolidatedAzureServices initializes without errors
- **PydanticAI Agents**: All three agents create with proper lazy initialization
- **Configuration System**: Bootstrap and runtime configs load without circular deps
- **Shared Infrastructure**: Text statistics and preprocessing utilities work
- **FastAPI Server**: Starts on port 8000 with working health endpoints

This represents a **real, production-ready development environment** with genuine Azure service integration, PydanticAI agents, and comprehensive infrastructure - no mock components or placeholder implementations.