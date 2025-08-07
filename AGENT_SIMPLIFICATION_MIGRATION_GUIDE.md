# Agent Architecture Simplification Migration Guide

This guide provides step-by-step instructions for migrating from the complex agent architecture to the simplified PydanticAI-compliant implementation.

## Overview of Changes

### Architecture Simplifications
- **Agent Creation**: Complex initialization → Direct `Agent()` instantiation
- **Dependencies**: Custom `Deps` classes → Simple, focused dependency models  
- **Tools**: `FunctionToolset` classes → Direct `@tool` functions
- **Configuration**: Multi-layer config loading → Environment variables + simple models
- **Orchestration**: Dual-graph workflows → Direct agent composition
- **Models**: 340+ models → ~50 essential models

### Key Benefits
- **75% reduction** in lines of code per agent
- **Eliminated complexity** in toolset registration and processor delegation
- **Simplified debugging** with direct tool functions
- **Better PydanticAI compliance** following official patterns
- **Cleaner testing** with focused dependencies

## Migration Steps

### Phase 1: Backup and Preparation

1. **Create backup of current implementation**:
   ```bash
   # Backup current agents
   cp -r agents/ agents_backup/
   
   # Backup current tests
   cp -r tests/ tests_backup/
   ```

2. **Verify current functionality** (baseline):
   ```bash
   # Run existing tests to establish baseline
   pytest tests/unit/test_agents_logic.py -v
   pytest tests/integration/test_agents_azure.py -v
   ```

### Phase 2: Agent-by-Agent Migration

#### Domain Intelligence Agent Migration

**Current Usage**:
```python
from agents.domain_intelligence.agent import get_domain_intelligence_agent
agent = get_domain_intelligence_agent()
result = await agent.run_async(prompt, deps=complex_deps)
```

**New Usage**:
```python
from agents.domain_intelligence.simplified_agent import get_domain_agent, DomainDeps
agent = get_domain_agent()
deps = DomainDeps(data_directory="/path/to/data")
result = await agent.run("Analyze domain", deps=deps)
```

**Breaking Changes**:
- `get_domain_intelligence_agent()` → `get_domain_agent()`
- Complex `DomainIntelligenceDeps` → Simple `DomainDeps` 
- `run_async()` → `run()`
- Result structure simplified to `DomainAnalysis` model

**Migration Script**:
```bash
# Replace import statements
find . -name "*.py" -exec sed -i 's/from agents.domain_intelligence.agent import get_domain_intelligence_agent/from agents.domain_intelligence.simplified_agent import get_domain_agent/g' {} +

# Replace function calls
find . -name "*.py" -exec sed -i 's/get_domain_intelligence_agent()/get_domain_agent()/g' {} +
```

#### Knowledge Extraction Agent Migration

**Current Usage**:
```python
from agents.knowledge_extraction.agent import get_knowledge_extraction_agent
from agents.knowledge_extraction.toolsets import KnowledgeExtractionDeps
agent = get_knowledge_extraction_agent()
result = await agent.run_async(prompt, deps=complex_deps)
```

**New Usage**:
```python
from agents.knowledge_extraction.simplified_agent import get_extraction_agent, ExtractionDeps, extract_knowledge
# Option 1: Direct agent usage
agent = get_extraction_agent()
deps = ExtractionDeps(confidence_threshold=0.8)
result = await agent.run("Extract knowledge from text", deps=deps)

# Option 2: Simple function (recommended for basic usage)
result = await extract_knowledge("text content", confidence_threshold=0.8)
```

**Breaking Changes**:
- `get_knowledge_extraction_agent()` → `get_extraction_agent()`
- Complex `KnowledgeExtractionDeps` → Simple `ExtractionDeps`
- Unified processor → Direct tool functions
- New simple function: `extract_knowledge()` for common use cases

#### Universal Search Agent Migration

**Current Usage**:
```python
from agents.universal_search.agent import get_universal_search_agent
from agents.universal_search.orchestrators import ConsolidatedSearchOrchestrator
agent = get_universal_search_agent()
orchestrator = ConsolidatedSearchOrchestrator()
```

**New Usage**:
```python
from agents.universal_search.simplified_agent import get_search_agent, SearchDeps, universal_search
# Option 1: Direct agent usage
agent = get_search_agent()
deps = SearchDeps(max_results=10, similarity_threshold=0.7)
result = await agent.run("Search for information", deps=deps)

# Option 2: Simple function (recommended)
result = await universal_search("query", max_results=10)
```

**Breaking Changes**:
- `get_universal_search_agent()` → `get_search_agent()`
- `ConsolidatedSearchOrchestrator` → Direct agent calls or `universal_search()`
- Complex search configuration → Simple `SearchDeps`
- New simple function: `universal_search()` for common use cases

### Phase 3: Orchestration Migration

#### Workflow Orchestration Changes

**Current Usage**:
```python
from agents.workflows.dual_graph_orchestrator import DualGraphOrchestrator
orchestrator = DualGraphOrchestrator()
await orchestrator.execute_config_extraction_pipeline(corpus_path)
await orchestrator.execute_search_workflow(query)
```

**New Usage**:
```python
from agents.simplified_orchestration import create_rag_orchestrator, analyze_corpus, search_knowledge, complete_rag

# Option 1: Orchestrator class
orchestrator = create_rag_orchestrator()
await orchestrator.process_document_corpus(corpus_path)
await orchestrator.execute_universal_search(query)

# Option 2: Simple functions (recommended)
corpus_result = await analyze_corpus(corpus_path)
search_result = await search_knowledge(query)
full_result = await complete_rag(corpus_path, query)
```

**Breaking Changes**:
- `DualGraphOrchestrator` → `SimplifiedRAGOrchestrator`
- Complex workflow graphs → Direct agent composition
- State management → Simple async/await coordination
- New simple functions for common patterns

### Phase 4: Configuration Migration

#### Model and Dependency Changes

**Current Dependencies**:
```python
from agents.core.data_models import (
    DomainIntelligenceDeps,
    KnowledgeExtractionDeps, 
    UniversalSearchDeps,
    ConsolidatedAzureServices
)
```

**New Dependencies**:
```python
from agents.domain_intelligence.simplified_agent import DomainDeps
from agents.knowledge_extraction.simplified_agent import ExtractionDeps
from agents.universal_search.simplified_agent import SearchDeps
# Note: ConsolidatedAzureServices no longer needed for basic usage
```

#### Configuration Loading Changes

**Current Configuration**:
```python
from config.centralized_config import get_model_config, get_extraction_config, get_search_config
model_config = get_model_config()
extraction_config = get_extraction_config(domain_name)
search_config = get_search_config()
```

**New Configuration**:
```python
import os
# Direct environment variable usage (no complex config loading)
model_deployment = os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
```

### Phase 5: Testing Migration

#### Test Structure Changes

**Current Test Pattern**:
```python
@pytest.mark.asyncio
async def test_complex_agent():
    from agents.domain_intelligence.agent import get_domain_intelligence_agent
    from agents.core.data_models import DomainIntelligenceDeps
    
    agent = get_domain_intelligence_agent()
    deps = DomainIntelligenceDeps(
        azure_services=mock_services,
        cache_manager=mock_cache,
        # ... complex setup
    )
    result = await agent.run_async(prompt, deps=deps)
```

**New Test Pattern**:
```python
@pytest.mark.asyncio
async def test_simplified_agent():
    from agents.domain_intelligence.simplified_agent import get_domain_agent, DomainDeps
    
    agent = get_domain_agent()
    deps = DomainDeps(data_directory="/test/data")
    result = await agent.run("Test prompt", deps=deps)
    
    assert result.data.detected_domain
    assert result.data.confidence > 0
```

#### Test Migration Script

```python
# Create new test files for simplified agents
# tests/unit/test_simplified_agents.py

import pytest
from agents.domain_intelligence.simplified_agent import get_domain_agent, DomainDeps
from agents.knowledge_extraction.simplified_agent import get_extraction_agent, extract_knowledge
from agents.universal_search.simplified_agent import get_search_agent, universal_search
from agents.simplified_orchestration import create_rag_orchestrator

@pytest.mark.asyncio
async def test_domain_agent():
    agent = get_domain_agent()
    deps = DomainDeps()
    result = await agent.run("Test domain analysis", deps=deps)
    assert result.data.detected_domain
    
@pytest.mark.asyncio  
async def test_extraction_agent():
    result = await extract_knowledge("Test text for extraction", confidence_threshold=0.6)
    assert result.entity_count >= 0
    
@pytest.mark.asyncio
async def test_search_agent():
    result = await universal_search("test query", max_results=5)
    assert result.total_results >= 0
    
@pytest.mark.asyncio
async def test_orchestration():
    orchestrator = create_rag_orchestrator()
    result = await orchestrator.execute_universal_search("test")
    assert result.success
```

## Validation and Rollback

### Pre-Migration Validation

1. **Run validation suite**:
   ```python
   from agents.validation_steps import validate_simplification
   import asyncio
   
   async def main():
       results = await validate_simplification()
       print(f"Validation success rate: {results['summary']['success_rate']:.1%}")
   
   asyncio.run(main())
   ```

2. **Performance comparison**:
   ```bash
   # Benchmark current implementation
   python -m pytest tests/performance/ --benchmark-only --benchmark-save=before
   
   # After migration
   python -m pytest tests/performance/ --benchmark-only --benchmark-save=after
   python -m pytest-benchmark compare before after
   ```

### Rollback Plan

If issues arise during migration:

1. **Immediate rollback**:
   ```bash
   # Restore from backup
   rm -rf agents/
   cp -r agents_backup/ agents/
   
   # Restore tests
   rm -rf tests/
   cp -r tests_backup/ tests/
   ```

2. **Gradual rollback**:
   ```python
   # Use compatibility imports during transition
   try:
       from agents.domain_intelligence.simplified_agent import get_domain_agent
   except ImportError:
       from agents.domain_intelligence.agent import get_domain_intelligence_agent as get_domain_agent
   ```

### Common Migration Issues

#### Issue 1: Model Import Errors
**Problem**: `ImportError` for simplified models
**Solution**: 
```python
# Add compatibility layer
try:
    from agents.domain_intelligence.simplified_agent import DomainDeps
except ImportError:
    from agents.core.data_models import DomainIntelligenceDeps as DomainDeps
```

#### Issue 2: Configuration Missing
**Problem**: Complex configuration not found
**Solution**:
```python
import os
# Use environment variables directly
deployment_name = os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o')
# Set reasonable defaults for missing config
```

#### Issue 3: Tool Function Changes
**Problem**: Custom toolset methods not found
**Solution**:
```python
# Replace toolset method calls with direct agent tools
# Old: await agent.toolsets[0].extract_entities(...)
# New: Use agent.run() with appropriate prompt
```

#### Issue 4: Dependency Injection Changes
**Problem**: `ConsolidatedAzureServices` not available
**Solution**:
```python
# For basic usage, dependencies are handled internally
# For advanced usage, create focused dependencies:
deps = ExtractionDeps(confidence_threshold=0.8)
# Instead of injecting complex service containers
```

## Success Criteria

Migration is successful when:

1. **All tests pass**: `pytest tests/ -v`
2. **Performance maintained**: Response times ≤ current implementation
3. **Functionality preserved**: All core features work as expected
4. **Code reduction**: ≥70% reduction in complexity metrics
5. **PydanticAI compliance**: Follows official PydanticAI patterns

## Timeline

- **Phase 1** (Backup): 1 day
- **Phase 2** (Agent Migration): 2-3 days
- **Phase 3** (Orchestration): 1-2 days  
- **Phase 4** (Configuration): 1 day
- **Phase 5** (Testing): 1-2 days
- **Total**: 6-9 days

## Support

For migration issues:

1. **Check validation results**: Run `agents/validation_steps.py`
2. **Review error patterns**: Common issues documented above
3. **Use rollback plan**: If critical issues arise
4. **Incremental migration**: Migrate one agent at a time if needed

This migration guide ensures a smooth transition to the simplified agent architecture while maintaining system functionality and improving code maintainability.