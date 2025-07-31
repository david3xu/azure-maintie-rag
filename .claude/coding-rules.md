# Critical Coding Rules

## FORBIDDEN Patterns ❌

### Hardcoded Values
```python
# ❌ WRONG
MEDICAL_ENTITIES = ["patient", "diagnosis", "treatment"]
MAINTENANCE_ACTIONS = ["repair", "replace", "inspect"]
if "medical" in domain_name:
    return self.medical_specific_processing()
```

### Fake Data and Placeholders
```python
# ❌ WRONG
def get_entities(self, text: str) -> List[str]:
    return ["entity1", "entity2", "placeholder_entity"]

async def search_documents(self, query: str) -> List[Document]:
    try:
        return await self.real_search(query)
    except Exception:
        return [Document(title="Sample Document", content="This is a sample...")]
```

### Direct Service Instantiation
```python
# ❌ WRONG
query_service = QueryService()  # Direct instantiation
workflow_service = WorkflowService()  # Bypasses DI
```

### Global State Anti-Pattern
```python
# ❌ WRONG
gnn_service = None  # Global variable
```

### Silent Failures
```python
# ❌ WRONG
try:
    return await self.real_processing(data)
except:
    return None  # Silent failure, no logging
```

## REQUIRED Patterns ✅

### Learn from Real Data
```python
# ✅ CORRECT
class EntityExtractor:
    async def extract_entities(self, text_corpus: List[str]) -> List[Entity]:
        entities = []
        for text in text_corpus:
            extracted = await self.nlp_service.analyze_text(text)
            entities.extend(extracted.entities)
        
        entity_patterns = self._analyze_entity_patterns(entities)
        return self._classify_entities_by_patterns(entity_patterns)
```

### Proper Dependency Injection
```python
# ✅ CORRECT
@router.post("/api/v1/query")
async def universal_query(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service)
):
    return await query_service.process_query(request)
```

### Async Operations
```python
# ✅ CORRECT
async def extract_entities_parallel(self, texts: List[str]) -> List[Entity]:
    extraction_tasks = [
        self._extract_from_single_text(text) 
        for text in texts
    ]
    results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

### Real Error Handling
```python
# ✅ CORRECT
async def process_document(self, document_path: str) -> ProcessingResult:
    try:
        document_content = await self.document_loader.load(document_path)
    except FileNotFoundError:
        raise DocumentNotFoundError(f"Document not found: {document_path}")
    except PermissionError:
        raise DocumentAccessError(f"Permission denied: {document_path}")
    
    return await self._process_content(document_content)
```

### Universal Design
```python
# ✅ CORRECT
class UniversalEntityProcessor:
    async def process_entities(self, text_corpus: List[str], domain_name: str):
        # Learn patterns from actual domain data
        learned_patterns = await self.pattern_learner.discover_patterns(text_corpus)
        
        # Apply learned patterns (not hardcoded rules)
        entities = []
        for text in text_corpus:
            text_entities = await self.nlp_service.extract_entities(
                text=text,
                patterns=learned_patterns,
                domain_context=domain_name
            )
            entities.extend(text_entities)
        
        return entities
```

## Architecture Rules

### 1. Tri-Modal Unity Principle
**Every feature must strengthen the unified search architecture**
- ✅ Enhance tri-modal search coordination
- ❌ Don't create competing search mechanisms

### 2. Data-Driven Domain Discovery
**All domain knowledge must be dynamically learned from raw text data**
- ✅ Extract patterns from actual text corpus
- ❌ No hardcoded domain assumptions or entity types

### 3. Async-First Performance Architecture
**All I/O operations must be asynchronous with parallel execution**
- ✅ Use asyncio.gather() for parallel operations
- ❌ No blocking synchronous operations

### 4. Azure-Native Service Integration
**Follow established Azure connection and authentication patterns**
- ✅ Use DefaultAzureCredential and service abstractions
- ❌ No direct Azure service instantiation in controllers

### 5. Observable Enterprise Architecture
**All components must include comprehensive monitoring and error handling**
- ✅ Structured logging with operation context
- ❌ No silent failures or generic error messages

### 6. Dependency Inversion and Testability
**All dependencies must be injected and components testable**
- ✅ Services depend on abstractions via DI
- ❌ No hard dependencies or tight coupling

## Performance Requirements
- **Response Time**: Sub-3-second processing (including agent reasoning)
- **Concurrency**: Support 100+ concurrent users
- **Accuracy**: 85-95% baseline, targeting 95-98% with agents
- **Scalability**: Unlimited domains with zero configuration

## Code Review Checklist
Before any code is merged, it must pass ALL these checks:

- [ ] Does this code learn from actual data instead of using hardcoded values?
- [ ] Are all return values based on real processing results?
- [ ] Is the implementation complete without TODO comments?
- [ ] Will this code work with any domain without modification?
- [ ] Are all I/O operations asynchronous?
- [ ] Does the code use proper dependency injection?
- [ ] Is there comprehensive error handling with context?
- [ ] Does the code meet sub-3-second response time requirements?