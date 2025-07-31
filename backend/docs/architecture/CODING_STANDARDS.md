# Azure Universal RAG - Coding Standards and Rules

## Executive Summary

This document establishes **mandatory coding standards** for the Azure Universal RAG with Intelligent Agents system. These rules ensure consistency, maintainability, and alignment with the core architectural principles of data-driven intelligence, universal scalability, and zero-configuration deployment.

## Core Principles

### **1. Data-Driven Everything**
**Every decision must be based on actual data, never assumptions or hardcoded values**

### **2. Universal Truth**  
**No fake values, placeholders, or fallbacks that don't represent real system state**

### **3. Zero Configuration**
**System must work with any domain from raw text data without manual configuration**

### **4. Production Reality**
**All code must be production-ready with real data handling, not demo approximations**

## Mandatory Coding Rules

### **Rule 1: Absolute Data-Driven Implementation**

#### **✅ REQUIRED: Always Use Real Data**
```python
# ✅ CORRECT - Learn from actual data
class EntityExtractor:
    async def extract_entities(self, text_corpus: List[str]) -> List[Entity]:
        """Extract entities from actual text using NLP analysis"""
        entities = []
        for text in text_corpus:
            # Use real NLP processing on actual text
            extracted = await self.nlp_service.analyze_text(text)
            entities.extend(extracted.entities)
        
        # Statistical analysis of discovered entities
        entity_patterns = self._analyze_entity_patterns(entities)
        return self._classify_entities_by_patterns(entity_patterns)

# ✅ CORRECT - Dynamic discovery from data
class DomainPatternDiscoverer:
    async def discover_domain_patterns(self, domain_name: str, raw_texts: List[str]):
        """Discover patterns from actual domain text corpus"""
        # Analyze actual text content
        content_analysis = await self.content_analyzer.analyze_corpus(raw_texts)
        
        # Extract real patterns from analysis
        return DomainConfig(
            name=domain_name,
            entity_types=content_analysis.discovered_entity_types,
            relationship_patterns=content_analysis.learned_relationships,
            confidence_score=content_analysis.statistical_confidence
        )
```

#### **❌ FORBIDDEN: Hardcoded Values and Assumptions**
```python
# ❌ WRONG - Hardcoded domain assumptions
MEDICAL_ENTITIES = ["patient", "diagnosis", "treatment"]  # FORBIDDEN
MAINTENANCE_ACTIONS = ["repair", "replace", "inspect"]   # FORBIDDEN

# ❌ WRONG - Assumption-based logic
if "medical" in domain_name:  # FORBIDDEN - Never assume domain characteristics
    return self.medical_specific_processing()

# ❌ WRONG - Hardcoded thresholds without data justification
if confidence_score > 0.8:  # FORBIDDEN - Where does 0.8 come from?
    return "high_confidence"
```

### **Rule 2: No Placeholders or Fake Data**

#### **✅ REQUIRED: Real Values or Explicit Null Handling**
```python
# ✅ CORRECT - Explicit null handling with real data validation
class AnalysisResult:
    def __init__(self, analysis_data: Optional[Dict[str, Any]]):
        if analysis_data is None:
            raise ValueError("Analysis data cannot be None - provide real analysis results")
        
        # Validate real data structure
        self.confidence = self._validate_confidence(analysis_data.get("confidence"))
        self.entities_found = analysis_data.get("entities", [])
        
        if not self.entities_found:
            logger.warning("No entities found in analysis - this may indicate data quality issues")
    
    def _validate_confidence(self, confidence: Optional[float]) -> float:
        """Validate confidence score from real analysis"""
        if confidence is None:
            raise ValueError("Confidence score missing from analysis results")
        
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid confidence score: {confidence}. Must be between 0.0 and 1.0")
        
        return confidence

# ✅ CORRECT - Real error handling without fake fallbacks
class SearchService:
    async def search(self, query: str, domain: str) -> SearchResult:
        """Execute search with real error handling"""
        try:
            # Attempt real search operation
            results = await self.azure_search_client.search(query, domain)
            
            if not results:
                # Real situation: no results found
                return SearchResult(
                    query=query,
                    results=[],
                    total_found=0,
                    message="No matching documents found for query",
                    search_successful=True  # Search worked, just no matches
                )
            
            return SearchResult(
                query=query,
                results=results,
                total_found=len(results),
                search_successful=True
            )
            
        except AzureSearchException as e:
            # Real error occurred - don't fake a response
            logger.error(f"Azure search failed for query '{query}': {e}")
            raise SearchServiceException(f"Search operation failed: {str(e)}") from e
```

#### **❌ FORBIDDEN: Placeholders and Fake Fallbacks**
```python
# ❌ WRONG - Placeholder values
def get_entities(self, text: str) -> List[str]:
    # FORBIDDEN - Never return fake placeholder data
    return ["entity1", "entity2", "placeholder_entity"]

# ❌ WRONG - Fake fallback data
async def search_documents(self, query: str) -> List[Document]:
    try:
        return await self.real_search(query)
    except Exception:
        # FORBIDDEN - Never return fake documents
        return [Document(title="Sample Document", content="This is a sample...")]

# ❌ WRONG - Fake confidence scores
def calculate_confidence(self, data: Any) -> float:
    # FORBIDDEN - Never return arbitrary confidence values
    return 0.85  # Where does this number come from?

# ❌ WRONG - TODO placeholders in production code
async def process_complex_query(self, query: str):
    # TODO: Implement proper logic  # FORBIDDEN in production code
    return {"status": "not_implemented"}
```

### **Rule 3: Production-Ready Implementation**

#### **✅ REQUIRED: Complete Implementation with Error Handling**
```python
# ✅ CORRECT - Complete implementation with comprehensive error handling
class GNNPredictionService:
    async def predict_relationships(self, entities: List[str], domain: str) -> PredictionResult:
        """Complete GNN prediction with real error handling and validation"""
        
        # Input validation
        if not entities:
            raise ValueError("Entity list cannot be empty for GNN prediction")
        
        if not domain:
            raise ValueError("Domain must be specified for GNN prediction")
        
        try:
            # Load real trained model for domain
            model = await self.model_registry.get_trained_model(domain)
            if model is None:
                raise ModelNotFoundError(f"No trained GNN model found for domain: {domain}")
            
            # Prepare real data for prediction  
            graph_data = await self.data_processor.prepare_prediction_data(entities, domain)
            
            # Execute real prediction
            predictions = await model.predict(graph_data)
            
            # Validate prediction results
            validated_predictions = self._validate_predictions(predictions, entities)
            
            return PredictionResult(
                entities=entities,
                domain=domain,
                predictions=validated_predictions,
                model_version=model.version,
                confidence_scores=predictions.confidence_scores,
                processing_time=predictions.execution_time
            )
            
        except ModelNotFoundError:
            # Real error - model doesn't exist for this domain
            raise
        except DataProcessingError as e:
            # Real error - data couldn't be processed
            logger.error(f"Data processing failed for GNN prediction: {e}")
            raise PredictionServiceError(f"Failed to process data for prediction: {str(e)}") from e
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error in GNN prediction: {e}", exc_info=True)
            raise PredictionServiceError(f"GNN prediction failed: {str(e)}") from e
    
    def _validate_predictions(self, predictions: Any, input_entities: List[str]) -> List[Prediction]:
        """Validate prediction results against input"""
        if not predictions:
            raise ValidationError("GNN model returned empty predictions")
        
        # Validate prediction structure and data quality
        validated = []
        for pred in predictions:
            if not self._is_valid_prediction(pred, input_entities):
                logger.warning(f"Invalid prediction filtered out: {pred}")
                continue
            validated.append(pred)
        
        if not validated:
            raise ValidationError("All GNN predictions were invalid - check model quality")
        
        return validated
```

#### **❌ FORBIDDEN: Incomplete or Mock Implementation**
```python
# ❌ WRONG - Mock/stub implementation
async def predict_relationships(self, entities: List[str]) -> List[str]:
    # FORBIDDEN - Never ship mock implementations
    return ["mock_relationship_1", "mock_relationship_2"]

# ❌ WRONG - Incomplete implementation with TODOs
async def complex_reasoning(self, query: str):
    # TODO: Implement actual reasoning logic  # FORBIDDEN
    # TODO: Add error handling                 # FORBIDDEN
    return {"result": "placeholder"}

# ❌ WRONG - Silent failures without proper error handling
async def search_operation(self, query: str):
    try:
        return await self.real_search(query)
    except:
        # FORBIDDEN - Silent failure, no logging, no proper error handling
        return []
```

### **Rule 4: Explicit Data Sources and Lineage**

#### **✅ REQUIRED: Document Data Sources and Transformations**
```python
# ✅ CORRECT - Clear data lineage and source documentation
class EntityAnalysisResult:
    """Results from entity analysis with complete data lineage"""
    
    def __init__(self, 
                 entities: List[Entity],
                 source_documents: List[str],
                 analysis_method: str,
                 processing_timestamp: datetime,
                 nlp_model_version: str):
        self.entities = entities
        self.data_lineage = DataLineage(
            source_documents=source_documents,
            processing_method=analysis_method,
            processed_at=processing_timestamp,
            nlp_model=nlp_model_version,
            entity_count=len(entities)
        )
    
    def get_source_evidence(self, entity: Entity) -> List[str]:
        """Get original text evidence for entity discovery"""
        return entity.source_evidence  # Real text snippets that led to entity identification

# ✅ CORRECT - Explicit confidence calculation
class ConfidenceCalculator:
    def calculate_entity_confidence(self, entity: Entity, analysis_context: AnalysisContext) -> float:
        """Calculate confidence based on explicit factors"""
        
        # Document frequency factor
        doc_frequency = entity.document_occurrences / analysis_context.total_documents
        
        # Context consistency factor  
        context_consistency = self._measure_context_consistency(entity, analysis_context)
        
        # NLP model confidence
        nlp_confidence = entity.nlp_extraction_confidence
        
        # Weighted combination with documented formula
        confidence = (
            (doc_frequency * 0.3) +
            (context_consistency * 0.4) + 
            (nlp_confidence * 0.3)
        )
        
        # Log calculation for auditability
        logger.debug(f"Confidence calculation for {entity.text}: "
                    f"doc_freq={doc_frequency:.3f}, "
                    f"context_consistency={context_consistency:.3f}, "
                    f"nlp_confidence={nlp_confidence:.3f}, "
                    f"final={confidence:.3f}")
        
        return confidence
```

#### **❌ FORBIDDEN: Unexplained Data Transformations**
```python
# ❌ WRONG - Mysterious confidence scores
def get_confidence(self, entity: str) -> float:
    # FORBIDDEN - Where does this number come from?
    return 0.75  # No explanation of calculation

# ❌ WRONG - Unexplained data transformations
def process_entities(self, raw_entities: List[str]) -> List[str]:
    # FORBIDDEN - What processing is happening?
    processed = []
    for entity in raw_entities:
        # Mystery processing with no documentation
        if len(entity) > 3:  # Why 3? Where does this threshold come from?
            processed.append(entity.upper())  # Why uppercase?
    return processed

# ❌ WRONG - Magic numbers without explanation
if similarity_score > 0.67:  # FORBIDDEN - Why 0.67?
    return "similar"
```

### **Rule 5: Universal Scalability**

#### **✅ REQUIRED: Domain-Agnostic Implementation**
```python
# ✅ CORRECT - Domain-agnostic with learned patterns
class UniversalEntityProcessor:
    async def process_entities(self, text_corpus: List[str], domain_name: str) -> ProcessingResult:
        """Process entities for any domain using learned patterns"""
        
        # Learn patterns from the actual domain data
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
        
        # Statistical validation using learned thresholds
        validated_entities = self._validate_entities_with_learned_thresholds(
            entities, learned_patterns.confidence_thresholds
        )
        
        return ProcessingResult(
            domain=domain_name,
            entities=validated_entities,
            patterns_used=learned_patterns,
            processing_method="universal_learned_patterns"
        )

# ✅ CORRECT - Configuration-driven behavior
class SearchOrchestrator:
    def __init__(self, domain_registry: DomainRegistry):
        self.domain_registry = domain_registry
    
    async def orchestrate_search(self, query: str, domain: str) -> SearchResult:
        """Orchestrate search using domain-specific learned configuration"""
        
        # Get learned configuration for this domain
        domain_config = await self.domain_registry.get_domain_config(domain)
        
        # Configure search based on learned patterns (not hardcoded)
        search_strategy = SearchStrategy(
            vector_weights=domain_config.learned_vector_weights,
            graph_traversal_depth=domain_config.optimal_traversal_depth,
            gnn_prediction_threshold=domain_config.learned_confidence_threshold
        )
        
        return await self._execute_configured_search(query, search_strategy)
```

#### **❌ FORBIDDEN: Domain-Specific Hardcoded Logic**
```python
# ❌ WRONG - Domain-specific hardcoded processing
def process_query(self, query: str, domain: str):
    # FORBIDDEN - Hardcoded domain logic
    if domain == "medical":
        return self._process_medical_query(query)
    elif domain == "legal":
        return self._process_legal_query(query)
    elif domain == "maintenance":
        return self._process_maintenance_query(query)
    else:
        return self._process_generic_query(query)  # Not truly universal

# ❌ WRONG - Hardcoded domain assumptions
class MedicalQueryProcessor:  # FORBIDDEN - Domain-specific classes
    def process(self, query: str):
        # Hardcoded medical logic
        if "patient" in query:
            return self.medical_specific_logic()

# ❌ WRONG - Hardcoded configuration per domain
DOMAIN_CONFIGS = {  # FORBIDDEN - Static configuration
    "medical": {"confidence_threshold": 0.8, "max_results": 20},
    "legal": {"confidence_threshold": 0.9, "max_results": 15},
    "maintenance": {"confidence_threshold": 0.7, "max_results": 25}
}
```

### **Rule 6: Comprehensive Error Handling**

#### **✅ REQUIRED: Explicit Error Handling with Context**
```python
# ✅ CORRECT - Comprehensive error handling with context
class DocumentProcessor:
    async def process_document(self, document_path: str, domain: str) -> ProcessingResult:
        """Process document with comprehensive error handling"""
        
        processing_context = ProcessingContext(
            document_path=document_path,
            domain=domain,
            started_at=datetime.utcnow()
        )
        
        try:
            # Validate inputs
            await self._validate_document_path(document_path)
            await self._validate_domain(domain)
            
            # Load document with specific error handling
            try:
                document_content = await self.document_loader.load(document_path)
            except FileNotFoundError:
                raise DocumentNotFoundError(f"Document not found: {document_path}")
            except PermissionError:
                raise DocumentAccessError(f"Permission denied accessing: {document_path}")
            except UnicodeDecodeError as e:
                raise DocumentFormatError(f"Cannot decode document {document_path}: {str(e)}")
            
            # Process with domain-specific error handling
            try:
                processing_result = await self._process_content(document_content, domain)
            except NLPProcessingError as e:
                raise ProcessingError(
                    f"NLP processing failed for {document_path} in domain {domain}: {str(e)}",
                    context=processing_context
                ) from e
            
            # Validate results
            self._validate_processing_result(processing_result, processing_context)
            
            return processing_result
            
        except DocumentProcessorError:
            # Re-raise our specific errors
            raise
        except Exception as e:
            # Catch unexpected errors with full context
            logger.error(
                f"Unexpected error processing document {document_path}",
                extra={
                    "document_path": document_path,
                    "domain": domain,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            raise ProcessingError(
                f"Unexpected error processing document: {str(e)}",
                context=processing_context
            ) from e
```

#### **❌ FORBIDDEN: Silent Failures and Generic Error Handling**
```python
# ❌ WRONG - Silent failure
async def process_data(self, data: Any):
    try:
        return await self.real_processing(data)
    except:
        # FORBIDDEN - Silent failure, no logging
        return None

# ❌ WRONG - Generic error handling
async def complex_operation(self, input_data: Any):
    try:
        return await self.process(input_data)
    except Exception as e:
        # FORBIDDEN - Too generic, no context, no specific handling
        logger.error(f"Error: {e}")
        return {"error": "Something went wrong"}

# ❌ WRONG - Swallowing important errors
async def critical_operation(self, data: Any):
    try:
        result = await self.important_processing(data)
    except ImportantError:
        # FORBIDDEN - Ignoring specific errors that should be handled
        pass  # This could hide critical issues
    
    return "completed"  # Lying about success
```

### **Rule 7: Performance and Scalability Requirements**

#### **✅ REQUIRED: Async-First with Performance Monitoring**
```python
# ✅ CORRECT - Async-first with performance tracking
class PerformanceTrackedService:
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    @trace_performance("entity_extraction")
    async def extract_entities_with_monitoring(self, texts: List[str], domain: str) -> ExtractionResult:
        """Extract entities with performance monitoring"""
        
        start_time = time.time()
        operation_id = f"extract_{domain}_{len(texts)}_docs"
        
        try:
            # Parallel processing for performance
            extraction_tasks = [
                self._extract_from_single_text(text, domain) 
                for text in texts
            ]
            
            # Execute in parallel with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*extraction_tasks, return_exceptions=True),
                timeout=30.0  # Real timeout based on performance requirements
            )
            
            # Process results and handle any exceptions
            successful_results = []
            failed_count = 0
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Entity extraction failed for one document: {result}")
                    failed_count += 1
                else:
                    successful_results.extend(result.entities)
            
            processing_time = time.time() - start_time
            
            # Record performance metrics
            self.metrics.record_operation_duration("entity_extraction", processing_time)
            self.metrics.record_throughput("entities_per_second", len(successful_results) / processing_time)
            self.metrics.record_error_rate("entity_extraction", failed_count / len(texts))
            
            # Validate performance requirements
            if processing_time > 5.0:  # Based on actual performance requirements
                logger.warning(f"Entity extraction took {processing_time:.2f}s, exceeding 5s target")
            
            return ExtractionResult(
                entities=successful_results,
                processing_time=processing_time,
                documents_processed=len(texts),
                documents_failed=failed_count,
                operation_id=operation_id
            )
            
        except asyncio.TimeoutError:
            self.metrics.increment_counter("entity_extraction_timeouts")
            raise PerformanceError(f"Entity extraction timed out after 30 seconds for {len(texts)} documents")
```

#### **❌ FORBIDDEN: Blocking Operations and Performance Ignorance**
```python
# ❌ WRONG - Blocking synchronous operations
def extract_entities(self, texts: List[str]) -> List[str]:
    # FORBIDDEN - Blocking loop
    results = []
    for text in texts:
        # FORBIDDEN - Synchronous processing of each item
        result = self.sync_nlp_process(text)  # Blocks entire system
        results.append(result)
    return results

# ❌ WRONG - No performance monitoring
async def process_large_dataset(self, data: List[Any]):
    # FORBIDDEN - No monitoring of performance
    results = []
    for item in data:
        result = await self.process_item(item)  # Sequential processing
        results.append(result)
    return results  # No timing, no metrics, no optimization

# ❌ WRONG - Ignoring performance requirements
async def slow_operation(self, input_data: Any):
    # FORBIDDEN - No consideration of response time requirements
    await asyncio.sleep(10)  # Arbitrary delay
    return "result"  # Violates sub-3-second requirement
```

## Code Review Checklist

### **Mandatory Review Questions**

Before any code is merged, it must pass ALL these checks:

#### **✅ Data-Driven Validation**
- [ ] Does this code learn from actual data instead of using hardcoded values?
- [ ] Are all configuration values derived from data analysis or explicit requirements?
- [ ] Is domain-specific logic replaced with universal patterns learned from data?

#### **✅ No Fake Data Validation**  
- [ ] Are all return values based on real processing results?
- [ ] Are there any placeholder values, fake fallbacks, or mock data?
- [ ] Do error conditions return honest error states instead of fake success?

#### **✅ Production Readiness Validation**
- [ ] Is the implementation complete without TODO comments?
- [ ] Are all error conditions properly handled with appropriate exceptions?
- [ ] Is there comprehensive logging for debugging and monitoring?

#### **✅ Universal Scalability Validation**
- [ ] Will this code work with any domain without modification?
- [ ] Are there any domain-specific assumptions or hardcoded logic?
- [ ] Does the code use learned patterns instead of domain-specific rules?

#### **✅ Performance Validation**
- [ ] Are all I/O operations asynchronous?
- [ ] Is there appropriate timeout handling for external operations?
- [ ] Are performance metrics collected for monitoring?
- [ ] Does the code meet the sub-3-second response time requirement?

## Enforcement

### **Automated Checks**
- [ ] **Linting Rules**: Custom rules to detect hardcoded values and placeholders
- [ ] **Type Checking**: Strict typing to prevent fake data propagation
- [ ] **Performance Tests**: Automated tests for response time requirements
- [ ] **Integration Tests**: Real data validation in CI/CD pipeline

### **Code Review Process**
- [ ] **Mandatory Reviews**: All code requires review against these standards
- [ ] **Architecture Review**: Complex changes require architecture team approval
- [ ] **Performance Review**: Changes affecting performance require performance team review

### **Consequences**
- [ ] **Blocking**: Code that violates these standards will not be merged
- [ ] **Refactoring**: Existing code that violates standards must be refactored
- [ ] **Documentation**: All exceptions must be documented with business justification

## Examples of Excellence

### **✅ Data-Driven Entity Recognition**
```python
class DataDrivenEntityRecognizer:
    """Example of excellent data-driven implementation"""
    
    async def recognize_entities(self, text_corpus: List[str], domain: str) -> RecognitionResult:
        # Learn patterns from actual text
        text_statistics = await self._analyze_text_statistics(text_corpus)
        entity_patterns = await self._discover_entity_patterns(text_statistics)
        
        # Use learned patterns for recognition
        recognized_entities = []
        for text in text_corpus:
            entities = await self._extract_with_learned_patterns(text, entity_patterns)
            recognized_entities.extend(entities)
        
        # Validate results using statistical confidence
        confidence_threshold = self._calculate_statistical_threshold(entity_patterns)
        validated_entities = [
            entity for entity in recognized_entities 
            if entity.confidence >= confidence_threshold
        ]
        
        return RecognitionResult(
            entities=validated_entities,
            patterns_learned=entity_patterns,
            confidence_threshold=confidence_threshold,
            statistical_basis=text_statistics
        )
```

### **✅ Universal Problem Solving**
```python
class UniversalProblemSolver:
    """Example of excellent universal implementation"""
    
    async def solve_problem(self, problem_description: str, domain: str) -> Solution:
        # Get learned problem-solving patterns for this domain
        domain_patterns = await self.domain_registry.get_patterns(domain)
        
        # Analyze problem using learned patterns (not hardcoded rules)
        problem_analysis = await self._analyze_problem_with_patterns(
            problem_description, domain_patterns
        )
        
        # Generate solution using tri-modal intelligence
        solution_candidates = await self._generate_solutions(problem_analysis)
        
        # Select best solution using learned effectiveness patterns
        best_solution = await self._select_optimal_solution(
            solution_candidates, domain_patterns.effectiveness_history
        )
        
        return Solution(
            problem=problem_description,
            solution=best_solution,
            confidence=best_solution.statistical_confidence,
            learning_basis=domain_patterns
        )
```

---

**These standards are mandatory for all code in the Azure Universal RAG system. They ensure we maintain the highest quality, performance, and architectural integrity while building a revolutionary intelligent system.**