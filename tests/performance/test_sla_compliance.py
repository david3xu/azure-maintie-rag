"""
SLA Compliance Tests - CODING_STANDARDS Compliant  
Validates sub-3-second performance requirements with real Azure services.
"""

import pytest
import time
import asyncio
from typing import Dict, Any, List


@pytest.mark.azure
@pytest.mark.performance
@pytest.mark.slow
class TestSLACompliance:
    """Test SLA compliance with real Azure services"""
    
    @pytest.mark.asyncio
    async def test_search_response_time_sla(self, universal_search_agent, performance_monitor):
        """Test search operations meet sub-3-second SLA"""
        
        test_queries = [
            "simple search query",
            "more complex search with multiple terms",
            "domain-specific technical query"
        ]
        
        sla_violations = []
        
        for query in test_queries:
            async with performance_monitor.measure_operation(f"search_sla_{hash(query) % 1000}", sla_target=3.0) as measurement:
                start_time = time.time()
                
                result = await universal_search_agent.process_query({
                    "query": query,
                    "limit": 10
                })
                
                duration = time.time() - start_time
                
                # Validate result exists
                assert result is not None
                
                # Track SLA compliance
                if duration > 3.0:
                    sla_violations.append({
                        "query": query,
                        "duration": duration,
                        "result": result
                    })
                
                print(f"⏱️  Query '{query[:30]}...': {duration:.3f}s "
                      f"({'✅ SLA OK' if duration <= 3.0 else '❌ SLA VIOLATION'})")
        
        # Report SLA compliance
        total_queries = len(test_queries)
        violations = len(sla_violations)
        compliance_rate = (total_queries - violations) / total_queries * 100
        
        print(f"📊 SLA Compliance: {compliance_rate:.1f}% ({total_queries - violations}/{total_queries})")
        
        # Fail if more than 10% SLA violations
        assert compliance_rate >= 90.0, f"SLA compliance too low: {compliance_rate:.1f}% (violations: {sla_violations})"
    
    @pytest.mark.asyncio
    async def test_knowledge_extraction_sla(self, knowledge_extraction_agent, sample_documents, performance_monitor):
        """Test knowledge extraction meets SLA requirements"""
        
        sla_violations = []
        
        for domain, document in sample_documents.items():
            async with performance_monitor.measure_operation(f"extraction_sla_{domain}", sla_target=3.0):
                start_time = time.time()
                
                result = await knowledge_extraction_agent.process_query({
                    "query": f"Extract knowledge from {domain} document",
                    "content": document,
                    "domain": domain
                })
                
                duration = time.time() - start_time
                
                # Validate result exists
                assert result is not None
                
                # Track SLA compliance
                if duration > 3.0:
                    sla_violations.append({
                        "domain": domain,
                        "duration": duration,
                        "document_length": len(document),
                        "result": result
                    })
                
                print(f"⏱️  {domain.title()} extraction: {duration:.3f}s "
                      f"({'✅ SLA OK' if duration <= 3.0 else '❌ SLA VIOLATION'})")
        
        # Report extraction SLA compliance
        total_extractions = len(sample_documents)
        violations = len(sla_violations)
        compliance_rate = (total_extractions - violations) / total_extractions * 100
        
        print(f"📊 Extraction SLA Compliance: {compliance_rate:.1f}% ({total_extractions - violations}/{total_extractions})")
        
        # Knowledge extraction is more complex, allow slightly lower compliance
        assert compliance_rate >= 80.0, f"Extraction SLA compliance too low: {compliance_rate:.1f}% (violations: {sla_violations})"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_sla(self, universal_search_agent, performance_monitor):
        """Test SLA compliance under concurrent load"""
        
        # Create multiple concurrent search operations
        concurrent_queries = [
            f"concurrent search query {i}"
            for i in range(5)  # 5 concurrent operations
        ]
        
        async def execute_search_with_timing(query: str, query_id: int):
            """Execute search with individual timing"""
            async with performance_monitor.measure_operation(f"concurrent_sla_{query_id}", sla_target=3.0):
                start_time = time.time()
                
                result = await universal_search_agent.process_query({
                    "query": query,
                    "limit": 5
                })
                
                duration = time.time() - start_time
                
                return {
                    "query_id": query_id,
                    "query": query,
                    "duration": duration,
                    "result": result,
                    "sla_compliant": duration <= 3.0
                }
        
        # Execute concurrent operations
        tasks = [
            execute_search_with_timing(query, i)
            for i, query in enumerate(concurrent_queries)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # Validate concurrent execution
        successful_operations = [
            result for result in results
            if isinstance(result, dict) and result.get("result") is not None
        ]
        
        assert len(successful_operations) > 0, f"All concurrent operations failed: {results}"
        
        # Check individual SLA compliance
        sla_compliant_ops = [
            op for op in successful_operations
            if op.get("sla_compliant", False)
        ]
        
        compliance_rate = len(sla_compliant_ops) / len(successful_operations) * 100
        
        print(f"🔄 Concurrent Operations Results:")
        print(f"   Total operations: {len(concurrent_queries)}")
        print(f"   Successful operations: {len(successful_operations)}")
        print(f"   SLA compliant: {len(sla_compliant_ops)}")
        print(f"   Compliance rate: {compliance_rate:.1f}%")
        print(f"   Total execution time: {total_duration:.3f}s")
        
        # Concurrent operations should maintain reasonable SLA compliance
        assert compliance_rate >= 70.0, f"Concurrent SLA compliance too low: {compliance_rate:.1f}%"
    
    @pytest.mark.asyncio
    async def test_azure_service_response_times(self, azure_services, performance_monitor):
        """Test individual Azure service response times"""
        
        service_timings = {}
        
        # Test Azure OpenAI response time
        if azure_services.openai_client:
            async with performance_monitor.measure_operation("azure_openai_sla", sla_target=2.0):
                start_time = time.time()
                
                result = await azure_services.openai_client.get_completion(
                    "Test prompt for timing",
                    max_tokens=20
                )
                
                duration = time.time() - start_time
                service_timings["openai"] = {
                    "duration": duration,
                    "sla_target": 2.0,
                    "compliant": duration <= 2.0,
                    "result_received": result is not None
                }
        
        # Test Azure Search response time
        if azure_services.search_client:
            async with performance_monitor.measure_operation("azure_search_sla", sla_target=1.0):
                start_time = time.time()
                
                result = await azure_services.search_client.search("test query", top=5)
                
                duration = time.time() - start_time
                service_timings["search"] = {
                    "duration": duration,
                    "sla_target": 1.0,
                    "compliant": duration <= 1.0,
                    "result_received": result is not None
                }
        
        # Report service timing results
        print(f"⚡ Azure Service Response Times:")
        for service, timing in service_timings.items():
            print(f"   {service.title()}: {timing['duration']:.3f}s "
                  f"(target: {timing['sla_target']}s, "
                  f"{'✅ OK' if timing['compliant'] else '❌ SLOW'})")
        
        # Validate at least one service meets its SLA
        compliant_services = [t for t in service_timings.values() if t["compliant"]]
        assert len(compliant_services) > 0, f"No Azure services meeting SLA: {service_timings}"


@pytest.mark.azure
@pytest.mark.performance
class TestPerformanceDegradation:
    """Test performance under various conditions"""
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self, knowledge_extraction_agent, performance_monitor):
        """Test performance with large documents"""
        
        # Create increasingly large documents
        base_text = "This is a test document for performance testing. " * 100
        document_sizes = [1, 5, 10]  # Multiples of base_text
        
        for size_multiplier in document_sizes:
            large_document = base_text * size_multiplier
            document_size_kb = len(large_document.encode('utf-8')) / 1024
            
            # Adjust SLA based on document size
            sla_target = min(3.0 + (size_multiplier - 1) * 0.5, 10.0)
            
            async with performance_monitor.measure_operation(
                f"large_doc_{size_multiplier}x", 
                sla_target=sla_target
            ):
                start_time = time.time()
                
                result = await knowledge_extraction_agent.process_query({
                    "query": "Extract knowledge from large document",
                    "content": large_document,
                    "domain": "general"
                })
                
                duration = time.time() - start_time
                
                assert result is not None
                
                print(f"📄 Large document ({document_size_kb:.1f}KB): {duration:.3f}s "
                      f"(SLA: {sla_target}s, {'✅ OK' if duration <= sla_target else '❌ SLOW'})")
    
    @pytest.mark.asyncio
    async def test_rapid_sequential_requests(self, universal_search_agent, performance_monitor):
        """Test performance under rapid sequential requests"""
        
        num_requests = 10
        rapid_queries = [f"rapid test query {i}" for i in range(num_requests)]
        
        individual_timings = []
        
        overall_start = time.time()
        
        for i, query in enumerate(rapid_queries):
            async with performance_monitor.measure_operation(f"rapid_{i}", sla_target=3.0):
                start_time = time.time()
                
                result = await universal_search_agent.process_query({
                    "query": query,
                    "limit": 3
                })
                
                duration = time.time() - start_time
                individual_timings.append(duration)
                
                assert result is not None
        
        overall_duration = time.time() - overall_start
        avg_duration = sum(individual_timings) / len(individual_timings)
        max_duration = max(individual_timings)
        
        print(f"🚀 Rapid Sequential Requests:")
        print(f"   Total requests: {num_requests}")
        print(f"   Overall time: {overall_duration:.3f}s")
        print(f"   Average per request: {avg_duration:.3f}s")
        print(f"   Maximum request time: {max_duration:.3f}s")
        print(f"   Requests per second: {num_requests / overall_duration:.1f}")
        
        # Validate performance doesn't degrade significantly
        assert avg_duration <= 3.5, f"Average request time too high: {avg_duration:.3f}s"
        assert max_duration <= 5.0, f"Maximum request time too high: {max_duration:.3f}s"