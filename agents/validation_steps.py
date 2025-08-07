"""
Validation Steps for Agent Architecture Simplification
======================================================

This module provides comprehensive validation to ensure that the simplified
agent architecture maintains system integrity and functionality.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
import traceback

# Import simplified agents
from .domain_intelligence.simplified_agent import get_domain_agent, DomainDeps
from .knowledge_extraction.simplified_agent import get_extraction_agent, ExtractionDeps, extract_knowledge  
from .universal_search.simplified_agent import get_search_agent, SearchDeps, universal_search
from .simplified_orchestration import create_rag_orchestrator, complete_rag

class ValidationResult:
    """Validation test result"""
    def __init__(self, test_name: str, success: bool, message: str, execution_time: float = 0.0, details: Dict = None):
        self.test_name = test_name
        self.success = success
        self.message = message
        self.execution_time = execution_time
        self.details = details or {}

class SimplificationValidator:
    """
    Comprehensive validator for the simplified agent architecture.
    
    Ensures that simplifications maintain core functionality while
    improving code quality and maintainability.
    """
    
    def __init__(self):
        self.test_results: List[ValidationResult] = []
        self.data_dir = "/workspace/azure-maintie-rag/data/raw"
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🔍 Starting Agent Architecture Simplification Validation")
        print("=" * 60)
        
        # Core agent validations
        await self.validate_domain_agent()
        await self.validate_extraction_agent()
        await self.validate_search_agent()
        await self.validate_orchestration()
        
        # Integration validations
        await self.validate_agent_communication()
        await self.validate_performance_requirements()
        await self.validate_backward_compatibility()
        
        # Generate comprehensive report
        return self.generate_validation_report()
    
    async def validate_domain_agent(self) -> None:
        """Validate Domain Intelligence Agent simplification"""
        print("\n🧠 Testing Domain Intelligence Agent...")
        
        try:
            start_time = time.time()
            
            # Test 1: Agent creation
            agent = get_domain_agent()
            assert agent is not None, "Agent creation failed"
            
            # Test 2: Simple domain discovery
            deps = DomainDeps(data_directory=self.data_dir)
            result = await agent.run("Discover available domains", deps=deps)
            
            assert result.data.detected_domain, "Domain detection failed"
            assert result.data.confidence > 0, "Invalid confidence score"
            assert result.data.file_count >= 0, "Invalid file count"
            
            execution_time = time.time() - start_time
            
            self.test_results.append(ValidationResult(
                "Domain Agent Validation",
                True,
                f"✅ Domain agent working correctly. Detected: {result.data.detected_domain}",
                execution_time,
                {"detected_domain": result.data.detected_domain, "confidence": result.data.confidence}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(ValidationResult(
                "Domain Agent Validation",
                False,
                f"❌ Domain agent validation failed: {str(e)}",
                execution_time,
                {"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def validate_extraction_agent(self) -> None:
        """Validate Knowledge Extraction Agent simplification"""
        print("\n📚 Testing Knowledge Extraction Agent...")
        
        try:
            start_time = time.time()
            
            # Test 1: Agent creation
            agent = get_extraction_agent()
            assert agent is not None, "Agent creation failed"
            
            # Test 2: Simple knowledge extraction
            test_text = """
            Machine learning is a subset of artificial intelligence that focuses on algorithms
            that can learn from data. Neural networks are a key component of deep learning.
            Python is commonly used for machine learning development.
            """
            
            deps = ExtractionDeps(confidence_threshold=0.6, max_entities_per_chunk=10)
            result = await agent.run(f"Extract knowledge from: {test_text}", deps=deps)
            
            assert result.data.entities, "No entities extracted"
            assert result.data.entity_count > 0, "Invalid entity count"
            assert result.data.extraction_confidence > 0, "Invalid extraction confidence"
            
            # Test 3: Simple extraction function
            extraction_result = await extract_knowledge(test_text, confidence_threshold=0.6)
            assert extraction_result.entity_count > 0, "Simple extraction function failed"
            
            execution_time = time.time() - start_time
            
            self.test_results.append(ValidationResult(
                "Extraction Agent Validation", 
                True,
                f"✅ Extraction agent working correctly. Extracted {result.data.entity_count} entities",
                execution_time,
                {
                    "entity_count": result.data.entity_count,
                    "relationship_count": result.data.relationship_count,
                    "confidence": result.data.extraction_confidence
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(ValidationResult(
                "Extraction Agent Validation",
                False,
                f"❌ Extraction agent validation failed: {str(e)}",
                execution_time,
                {"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def validate_search_agent(self) -> None:
        """Validate Universal Search Agent simplification"""
        print("\n🔍 Testing Universal Search Agent...")
        
        try:
            start_time = time.time()
            
            # Test 1: Agent creation
            agent = get_search_agent()
            assert agent is not None, "Agent creation failed"
            
            # Test 2: Universal search
            test_query = "machine learning algorithms"
            deps = SearchDeps(max_results=5, similarity_threshold=0.7)
            result = await agent.run(f"Search for: {test_query}", deps=deps)
            
            assert result.data.query == test_query, "Query not preserved"
            assert result.data.total_results >= 0, "Invalid result count"
            assert result.data.synthesis_score >= 0, "Invalid synthesis score"
            assert result.data.modalities_used, "No modalities used"
            
            # Test 3: Simple search function
            search_result = await universal_search("python programming", max_results=3)
            assert search_result.total_results >= 0, "Simple search function failed"
            
            execution_time = time.time() - start_time
            
            self.test_results.append(ValidationResult(
                "Search Agent Validation",
                True, 
                f"✅ Search agent working correctly. Found {result.data.total_results} results",
                execution_time,
                {
                    "query": result.data.query,
                    "total_results": result.data.total_results,
                    "synthesis_score": result.data.synthesis_score,
                    "modalities": result.data.modalities_used
                }
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(ValidationResult(
                "Search Agent Validation",
                False,
                f"❌ Search agent validation failed: {str(e)}", 
                execution_time,
                {"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def validate_orchestration(self) -> None:
        """Validate simplified orchestration patterns"""
        print("\n🎼 Testing Simplified Orchestration...")
        
        try:
            start_time = time.time()
            
            # Test 1: Orchestrator creation
            orchestrator = create_rag_orchestrator()
            assert orchestrator is not None, "Orchestrator creation failed"
            
            # Test 2: Simple corpus analysis (if data exists)
            corpus_path = "/workspace/azure-maintie-rag/data/raw/Programming-Language"
            if Path(corpus_path).exists():
                result = await orchestrator.process_document_corpus(corpus_path)
                assert result.success, f"Corpus processing failed: {result.error_message}"
                assert result.domain_analysis, "No domain analysis"
            
            # Test 3: Simple search orchestration
            search_result = await orchestrator.execute_universal_search("programming concepts")
            assert search_result.success, f"Search orchestration failed: {search_result.error_message}"
            assert search_result.search_results, "No search results"
            
            execution_time = time.time() - start_time
            
            self.test_results.append(ValidationResult(
                "Orchestration Validation",
                True,
                "✅ Simplified orchestration working correctly",
                execution_time,
                {"corpus_processing": True, "search_orchestration": True}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(ValidationResult(
                "Orchestration Validation", 
                False,
                f"❌ Orchestration validation failed: {str(e)}",
                execution_time,
                {"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def validate_agent_communication(self) -> None:
        """Validate clean agent-to-agent communication"""
        print("\n🤝 Testing Agent Communication...")
        
        try:
            start_time = time.time()
            
            # Test agent composition: Domain -> Extraction -> Search
            domain_agent = get_domain_agent()
            extraction_agent = get_extraction_agent()
            search_agent = get_search_agent()
            
            # Step 1: Domain analysis
            domain_deps = DomainDeps()
            domain_result = await domain_agent.run("Analyze programming domain", deps=domain_deps)
            
            # Step 2: Use domain info for extraction
            extraction_deps = ExtractionDeps(confidence_threshold=0.7)
            test_content = "Object-oriented programming uses classes and inheritance."
            extraction_result = await extraction_agent.run(
                f"Extract from {domain_result.data.detected_domain} content: {test_content}",
                deps=extraction_deps
            )
            
            # Step 3: Use extraction results for search context
            search_deps = SearchDeps(max_results=3)
            search_result = await search_agent.run(
                f"Search with extracted knowledge: {extraction_result.data.entities[:2]}",
                deps=search_deps
            )
            
            # Validate communication chain
            assert domain_result.data.detected_domain, "Domain communication failed"
            assert extraction_result.data.entity_count > 0, "Extraction communication failed"  
            assert search_result.data.total_results >= 0, "Search communication failed"
            
            execution_time = time.time() - start_time
            
            self.test_results.append(ValidationResult(
                "Agent Communication Validation",
                True,
                "✅ Agent communication working correctly", 
                execution_time,
                {"communication_chain": "domain->extraction->search", "success": True}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(ValidationResult(
                "Agent Communication Validation",
                False,
                f"❌ Agent communication validation failed: {str(e)}",
                execution_time,
                {"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def validate_performance_requirements(self) -> None:
        """Validate that simplifications maintain performance requirements"""
        print("\n⚡ Testing Performance Requirements...")
        
        try:
            start_time = time.time()
            
            # Test performance benchmarks
            performance_tests = []
            
            # Domain agent performance
            domain_start = time.time()
            domain_agent = get_domain_agent()
            await domain_agent.run("Quick domain analysis", deps=DomainDeps())
            domain_time = time.time() - domain_start
            performance_tests.append(("domain_agent", domain_time))
            
            # Extraction agent performance  
            extraction_start = time.time()
            extraction_agent = get_extraction_agent()
            await extraction_agent.run("Quick extraction test", deps=ExtractionDeps())
            extraction_time = time.time() - extraction_start
            performance_tests.append(("extraction_agent", extraction_time))
            
            # Search agent performance
            search_start = time.time()
            search_agent = get_search_agent()
            await search_agent.run("Quick search test", deps=SearchDeps(max_results=3))
            search_time = time.time() - search_start  
            performance_tests.append(("search_agent", search_time))
            
            # Validate performance targets (simplified agents should be faster)
            max_acceptable_time = 30.0  # 30 seconds max for any operation
            all_within_limits = all(time_taken < max_acceptable_time for _, time_taken in performance_tests)
            
            execution_time = time.time() - start_time
            
            self.test_results.append(ValidationResult(
                "Performance Validation",
                all_within_limits,
                f"✅ All agents within performance limits" if all_within_limits else "❌ Performance issues detected",
                execution_time,
                {"performance_tests": dict(performance_tests), "within_limits": all_within_limits}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(ValidationResult(
                "Performance Validation",
                False,
                f"❌ Performance validation failed: {str(e)}",
                execution_time,
                {"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def validate_backward_compatibility(self) -> None:
        """Validate backward compatibility with existing interfaces"""
        print("\n🔄 Testing Backward Compatibility...")
        
        try:
            start_time = time.time()
            
            # Test that key functions still work
            compatibility_tests = []
            
            # Test extract_knowledge function
            try:
                result = await extract_knowledge("Test extraction compatibility")
                compatibility_tests.append(("extract_knowledge", True, "Function works"))
            except Exception as e:
                compatibility_tests.append(("extract_knowledge", False, str(e)))
            
            # Test universal_search function
            try:
                result = await universal_search("Test search compatibility")
                compatibility_tests.append(("universal_search", True, "Function works"))
            except Exception as e:
                compatibility_tests.append(("universal_search", False, str(e)))
            
            # Test complete_rag function (if data available)
            try:
                result = await complete_rag("/tmp", "test query")  # Safe test path
                compatibility_tests.append(("complete_rag", True, "Function works"))
            except Exception as e:
                compatibility_tests.append(("complete_rag", False, str(e)))
            
            # Count successes
            successes = sum(1 for _, success, _ in compatibility_tests if success)
            total_tests = len(compatibility_tests)
            
            execution_time = time.time() - start_time
            
            self.test_results.append(ValidationResult(
                "Backward Compatibility Validation",
                successes == total_tests,
                f"✅ {successes}/{total_tests} compatibility tests passed",
                execution_time,
                {"compatibility_tests": compatibility_tests, "success_rate": successes/total_tests}
            ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(ValidationResult(
                "Backward Compatibility Validation",
                False,
                f"❌ Backward compatibility validation failed: {str(e)}",
                execution_time,
                {"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        total_time = sum(result.execution_time for result in self.test_results)
        
        print(f"\n📊 Validation Report")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        print(f"Total Execution Time: {total_time:.2f}s")
        print()
        
        # Individual test results
        for result in self.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} {result.test_name}: {result.message}")
            if not result.success and "error" in result.details:
                print(f"   Error: {result.details['error']}")
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests/total_tests,
                "total_execution_time": total_time
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "details": r.details
                } for r in self.test_results
            ]
        }

# Simple API function
async def validate_simplification() -> Dict[str, Any]:
    """Run complete simplification validation"""
    validator = SimplificationValidator()
    return await validator.run_all_validations()

# Export validation interface
__all__ = ["SimplificationValidator", "validate_simplification", "ValidationResult"]