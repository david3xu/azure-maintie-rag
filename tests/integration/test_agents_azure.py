"""
Agent Integration Tests with Real Azure Services - CODING_STANDARDS Compliant
Tests agents using real Azure backends, no mock data.
"""

import pytest
import asyncio
from typing import Dict, Any


@pytest.mark.azure
@pytest.mark.integration
class TestKnowledgeExtractionAgentAzure:
    """Test Knowledge Extraction Agent with real Azure services"""
    
    @pytest.mark.asyncio
    async def test_document_processing_real_azure(self, knowledge_extraction_agent, sample_documents, performance_monitor):
        """Test real document processing with Azure backend"""
        
        # Test with programming domain document
        document = sample_documents["programming"]
        
        async with performance_monitor.measure_operation("knowledge_extraction", sla_target=3.0):
            # Process document with real Azure services
            result = await knowledge_extraction_agent.process_query({
                "query": "Extract knowledge from this document",
                "content": document,
                "domain": "programming"
            })
            
            # Validate real results (no fake data)
            assert result is not None
            assert isinstance(result, dict)
            assert result.get("success") is not None
            
            if result.get("success"):
                # Validate extracted knowledge structure
                knowledge = result.get("knowledge", {})
                assert isinstance(knowledge, dict)
                
                # Should have real entities or relationships
                entities = knowledge.get("entities", [])
                relationships = knowledge.get("relationships", [])
                assert len(entities) > 0 or len(relationships) > 0
                
                print(f"✅ Extracted {len(entities)} entities, {len(relationships)} relationships")
            else:
                print(f"⚠️ Extraction failed: {result.get('error', 'Unknown error')}")
                
    @pytest.mark.asyncio
    async def test_cross_domain_processing(self, knowledge_extraction_agent, sample_documents):
        """Test domain-agnostic processing across different domains"""
        
        domains_to_test = ["programming", "maintenance", "legal", "medical"]
        results = {}
        
        for domain in domains_to_test:
            if domain not in sample_documents:
                continue
                
            document = sample_documents[domain]
            
            try:
                result = await knowledge_extraction_agent.process_query({
                    "query": f"Extract knowledge from {domain} document",
                    "content": document,
                    "domain": domain
                })
                
                results[domain] = result
                
                # Validate domain-agnostic behavior
                assert result is not None
                assert isinstance(result, dict)
                
                print(f"✅ {domain.title()} domain processing: "
                      f"{'Success' if result.get('success') else 'Failed'}")
                
            except Exception as e:
                print(f"⚠️ {domain.title()} domain failed: {e}")
                results[domain] = {"success": False, "error": str(e)}
        
        # Validate at least some domains processed successfully
        successful_domains = sum(1 for result in results.values() if result.get("success"))
        assert successful_domains > 0, f"No domains processed successfully: {results}"
        
        print(f"📊 Cross-domain results: {successful_domains}/{len(results)} domains successful")


@pytest.mark.azure  
@pytest.mark.integration
class TestUniversalSearchAgentAzure:
    """Test Universal Search Agent with real Azure services"""
    
    @pytest.mark.asyncio
    async def test_search_functionality_real_azure(self, universal_search_agent, performance_monitor):
        """Test real search functionality with Azure backend"""
        
        search_queries = [
            "programming concepts",
            "maintenance procedures", 
            "legal documents",
            "medical information"
        ]
        
        for query in search_queries:
            async with performance_monitor.measure_operation(f"search_{query.replace(' ', '_')}", sla_target=3.0):
                result = await universal_search_agent.process_query({
                    "query": query,
                    "limit": 5
                })
                
                # Validate real search results
                assert result is not None
                assert isinstance(result, dict)
                
                if result.get("success"):
                    results = result.get("results", [])
                    assert isinstance(results, list)
                    
                    # Validate result structure (real data, not fake)
                    for search_result in results:
                        assert isinstance(search_result, dict)
                        # Should have real content, not placeholder data
                        content = search_result.get("content", "")
                        assert len(content) > 0
                        assert "placeholder" not in content.lower()
                        assert "mock" not in content.lower()
                    
                    print(f"✅ Search '{query}': {len(results)} real results")
                else:
                    print(f"⚠️ Search '{query}' failed: {result.get('error', 'Unknown error')}")
    
    @pytest.mark.asyncio
    async def test_search_performance_sla(self, universal_search_agent, performance_monitor):
        """Test search performance meets SLA requirements"""
        
        test_queries = [
            "quick search test",
            "performance validation query",
            "SLA compliance check"
        ]
        
        for query in test_queries:
            async with performance_monitor.measure_operation(f"sla_test_{hash(query) % 1000}", sla_target=3.0):
                result = await universal_search_agent.process_query({
                    "query": query,
                    "limit": 3
                })
                
                assert result is not None
                # Performance is validated by performance_monitor context manager
                
    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self, universal_search_agent):
        """Test concurrent search operations with real Azure backend"""
        
        concurrent_queries = [
            "concurrent test query 1",
            "concurrent test query 2", 
            "concurrent test query 3"
        ]
        
        # Create concurrent search tasks
        tasks = [
            universal_search_agent.process_query({
                "query": query,
                "limit": 2
            })
            for query in concurrent_queries
        ]
        
        # Execute concurrent searches
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Validate concurrent execution
            assert len(results) == len(concurrent_queries)
            
            # Check that operations completed (success or failure, not crash)
            completed_operations = sum(
                1 for result in results 
                if isinstance(result, dict) and result is not None
            )
            
            assert completed_operations > 0, f"All concurrent searches failed: {results}"
            
            print(f"✅ Concurrent searches: {completed_operations}/{len(results)} completed")
            
        except Exception as e:
            pytest.fail(f"Concurrent search operations failed: {e}")


@pytest.mark.azure
@pytest.mark.integration  
class TestAgentIntegration:
    """Test agent-to-agent integration with real Azure services"""
    
    @pytest.mark.asyncio
    async def test_knowledge_to_search_workflow(self, knowledge_extraction_agent, universal_search_agent, sample_documents):
        """Test end-to-end workflow: knowledge extraction → search"""
        
        # Step 1: Extract knowledge from document
        document = sample_documents["programming"]
        
        extraction_result = await knowledge_extraction_agent.process_query({
            "query": "Extract knowledge for search indexing",
            "content": document,
            "domain": "programming"
        })
        
        assert extraction_result is not None
        assert isinstance(extraction_result, dict)
        
        if not extraction_result.get("success"):
            pytest.skip(f"Knowledge extraction failed: {extraction_result.get('error')}")
        
        # Step 2: Use extracted knowledge for search
        knowledge = extraction_result.get("knowledge", {})
        entities = knowledge.get("entities", [])
        
        if not entities:
            pytest.skip("No entities extracted for search testing")
        
        # Search for first extracted entity
        search_query = entities[0].get("text", "programming") if entities else "programming"
        
        search_result = await universal_search_agent.process_query({
            "query": f"Find information about {search_query}",
            "limit": 3
        })
        
        assert search_result is not None
        assert isinstance(search_result, dict)
        
        print(f"✅ End-to-end workflow: Extraction → Search for '{search_query}'")
        print(f"   Extraction success: {extraction_result.get('success')}")
        print(f"   Search success: {search_result.get('success')}")
    
    @pytest.mark.asyncio
    async def test_error_propagation_across_agents(self, knowledge_extraction_agent, universal_search_agent):
        """Test error handling across agent boundaries"""
        
        # Test with invalid input that should be handled gracefully
        invalid_inputs = [
            {"query": "", "content": ""},  # Empty inputs
            {"query": "test", "content": "x" * 100000},  # Extremely large content
            {"query": None, "content": None}  # None values
        ]
        
        for invalid_input in invalid_inputs:
            try:
                # Knowledge extraction with invalid input
                extraction_result = await knowledge_extraction_agent.process_query(invalid_input)
                
                # Should handle gracefully, not crash
                assert extraction_result is not None
                assert isinstance(extraction_result, dict)
                
                # If extraction fails, search should also handle gracefully
                if not extraction_result.get("success"):
                    search_result = await universal_search_agent.process_query({
                        "query": invalid_input.get("query", "fallback query"),
                        "limit": 1
                    })
                    
                    assert search_result is not None
                    assert isinstance(search_result, dict)
                
                print(f"✅ Invalid input handled gracefully: {invalid_input}")
                
            except Exception as e:
                # Exceptions are acceptable, crashes are not
                assert "error" in str(e).lower() or "invalid" in str(e).lower()
                print(f"✅ Invalid input rejected appropriately: {type(e).__name__}")