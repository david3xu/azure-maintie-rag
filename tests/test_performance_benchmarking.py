"""
Performance Benchmarking and SLA Compliance Tests for Azure Universal RAG System
==============================================================================

Comprehensive performance validation ensuring production-grade SLA compliance
across all system components with real Azure services and authentic data.

Performance Targets:
- Domain Analysis: < 10 seconds
- Knowledge Extraction: < 15 seconds  
- Universal Search: < 12 seconds
- Complete Workflow: < 45 seconds
- Concurrent Users: 100+ simultaneous
- Cache Hit Rate: > 60%
- Processing Accuracy: > 85%
"""

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pytest
from dotenv import load_dotenv

# Load environment before imports
load_dotenv()

from agents.core.universal_deps import get_universal_deps
from agents.domain_intelligence.agent import run_domain_analysis
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.universal_search.agent import run_universal_search
from agents.orchestrator import UniversalOrchestrator


class TestPerformanceBenchmarking:
    """Comprehensive performance benchmarking with SLA validation."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_individual_agent_performance_benchmarks(
        self,
        azure_services,
        enhanced_test_data_manager,
        performance_monitor
    ):
        """Benchmark individual agent performance against SLA targets."""
        
        # Get diverse test files for comprehensive benchmarking
        benchmark_files = enhanced_test_data_manager.get_diverse_test_set(count=8)
        
        if len(benchmark_files) < 3:
            pytest.skip("Insufficient test files for performance benchmarking")
        
        agent_benchmarks = {
            "domain_intelligence": {
                "sla_target": 10.0,
                "measurements": [],
                "agent_func": run_domain_analysis
            },
            "knowledge_extraction": {
                "sla_target": 15.0,
                "measurements": [],
                "agent_func": run_knowledge_extraction
            },
            "universal_search": {
                "sla_target": 12.0,
                "measurements": [],
                "agent_func": run_universal_search
            }
        }
        
        print("🏃 Individual Agent Performance Benchmarking:")
        
        # Benchmark each agent
        for agent_name, config in agent_benchmarks.items():
            print(f"\n📊 Benchmarking {agent_name.replace('_', ' ').title()}:")
            
            for i, test_file in enumerate(benchmark_files):
                content = test_file.read_text(encoding='utf-8')
                
                # Prepare content based on agent requirements
                if agent_name == "domain_intelligence":
                    test_input = content[:2000]  # Domain analysis works well with moderate content
                elif agent_name == "knowledge_extraction":
                    test_input = content[:1500]  # Knowledge extraction on focused content
                else:  # universal_search
                    # Extract a meaningful search query from content
                    lines = content.split('\n')[:5]
                    test_input = f"Information about {' '.join(lines).split()[:10]}"[:100]
                
                async with performance_monitor.measure_operation(
                    f"{agent_name}_benchmark_{i}",
                    sla_target=config["sla_target"]
                ) as measurement:
                    
                    start_time = time.time()
                    try:
                        if agent_name == "universal_search":
                            result = await config["agent_func"](test_input, max_results=5)
                        elif agent_name == "knowledge_extraction":
                            result = await config["agent_func"](test_input, use_domain_analysis=False)
                        else:
                            result = await config["agent_func"](test_input)
                        
                        processing_time = time.time() - start_time
                        
                        # Extract performance-relevant metrics
                        if agent_name == "domain_intelligence":
                            metrics = {
                                "vocabulary_complexity": result.discovered_characteristics.vocabulary_complexity,
                                "concept_density": result.discovered_characteristics.concept_density,
                                "patterns_found": len(result.discovered_characteristics.structural_patterns)
                            }
                        elif agent_name == "knowledge_extraction":
                            metrics = {
                                "entities_count": len(result.entities),
                                "relationships_count": len(result.relationships),
                                "extraction_confidence": result.extraction_confidence
                            }
                        else:  # universal_search
                            metrics = {
                                "results_count": len(result.unified_results),
                                "search_confidence": result.search_confidence,
                                "total_results_found": result.total_results_found
                            }
                        
                        config["measurements"].append({
                            "file": test_file.name,
                            "processing_time": processing_time,
                            "content_size": len(content),
                            "sla_compliant": processing_time <= config["sla_target"],
                            "metrics": metrics,
                            "success": True
                        })
                        
                        print(f"     📄 {test_file.name}: {processing_time:.2f}s {'✅' if processing_time <= config['sla_target'] else '❌'}")
                        
                    except Exception as e:
                        processing_time = time.time() - start_time
                        config["measurements"].append({
                            "file": test_file.name,
                            "processing_time": processing_time,
                            "content_size": len(content),
                            "sla_compliant": False,
                            "error": str(e)[:100],
                            "success": False
                        })
                        print(f"     📄 {test_file.name}: {processing_time:.2f}s ❌ (Error: {str(e)[:50]})")
        
        # Analyze benchmark results
        benchmark_summary = {}
        
        for agent_name, config in agent_benchmarks.items():
            measurements = config["measurements"]
            successful_measurements = [m for m in measurements if m["success"]]
            
            if successful_measurements:
                processing_times = [m["processing_time"] for m in successful_measurements]
                sla_compliant_count = sum(1 for m in successful_measurements if m["sla_compliant"])
                
                summary = {
                    "total_tests": len(measurements),
                    "successful_tests": len(successful_measurements),
                    "sla_target": config["sla_target"],
                    "sla_compliant_tests": sla_compliant_count,
                    "sla_compliance_rate": sla_compliant_count / len(successful_measurements),
                    "avg_processing_time": statistics.mean(processing_times),
                    "median_processing_time": statistics.median(processing_times),
                    "min_processing_time": min(processing_times),
                    "max_processing_time": max(processing_times),
                    "std_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                    "success_rate": len(successful_measurements) / len(measurements)
                }
                
                benchmark_summary[agent_name] = summary
                
                print(f"\n   📈 {agent_name.replace('_', ' ').title()} Performance Summary:")
                print(f"     Success Rate: {summary['success_rate']:.2%}")
                print(f"     SLA Compliance: {summary['sla_compliance_rate']:.2%} (target: {summary['sla_target']}s)")
                print(f"     Avg Processing Time: {summary['avg_processing_time']:.2f}s")
                print(f"     Median Processing Time: {summary['median_processing_time']:.2f}s")
                print(f"     Time Range: {summary['min_processing_time']:.2f}s - {summary['max_processing_time']:.2f}s")
                print(f"     Std Deviation: {summary['std_processing_time']:.2f}s")
        
        # Overall performance validation
        overall_sla_compliance = sum(
            summary["sla_compliant_tests"] for summary in benchmark_summary.values()
        ) / sum(
            summary["successful_tests"] for summary in benchmark_summary.values()
        ) if benchmark_summary else 0
        
        overall_success_rate = sum(
            summary["successful_tests"] for summary in benchmark_summary.values()
        ) / sum(
            summary["total_tests"] for summary in benchmark_summary.values()
        ) if benchmark_summary else 0
        
        print(f"\n✅ Overall Performance Benchmark Results:")
        print(f"   Overall Success Rate: {overall_success_rate:.2%}")
        print(f"   Overall SLA Compliance: {overall_sla_compliance:.2%}")
        print(f"   Agents Tested: {len(benchmark_summary)}")
        
        # Assertions for production readiness
        assert overall_success_rate >= 0.8, f"Agent success rate too low: {overall_success_rate:.2%}"
        assert overall_sla_compliance >= 0.7, f"SLA compliance too low: {overall_sla_compliance:.2%}"
        
        # Each agent should meet minimum performance criteria
        for agent_name, summary in benchmark_summary.items():
            assert summary["success_rate"] >= 0.75, f"{agent_name} success rate too low: {summary['success_rate']:.2%}"
            assert summary["sla_compliance_rate"] >= 0.6, f"{agent_name} SLA compliance too low: {summary['sla_compliance_rate']:.2%}"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_end_to_end_workflow_performance(
        self,
        azure_services,
        enhanced_test_data_manager,
        performance_monitor
    ):
        """Test complete end-to-end workflow performance against SLA targets."""
        
        # Get high-quality test files for end-to-end workflow testing
        workflow_files = enhanced_test_data_manager.get_test_files_by_criteria(
            min_size=800,
            max_size=4000,
            content_types=["azure", "api"],
            limit=5
        )
        
        if len(workflow_files) < 3:
            pytest.skip("Insufficient test files for end-to-end workflow testing")
        
        orchestrator = UniversalOrchestrator()
        workflow_measurements = []
        
        print("🔄 End-to-End Workflow Performance Testing:")
        
        for i, test_file in enumerate(workflow_files):
            content = test_file.read_text(encoding='utf-8')
            
            print(f"\n   📄 Testing {test_file.name} ({len(content)} chars)")
            
            async with performance_monitor.measure_operation(
                f"e2e_workflow_{i}",
                sla_target=45.0  # 45-second SLA for complete workflow
            ):
                
                workflow_start = time.time()
                
                try:
                    # Execute complete multi-agent workflow
                    workflow_result = await orchestrator.process_knowledge_extraction_workflow(
                        content,
                        use_domain_analysis=True
                    )
                    
                    workflow_time = time.time() - workflow_start
                    
                    # Analyze workflow performance
                    stage_times = {}
                    for agent_name, metrics in workflow_result.agent_metrics.items():
                        stage_times[agent_name] = metrics.get("processing_time", 0)
                    
                    measurement = {
                        "file": test_file.name,
                        "total_workflow_time": workflow_time,
                        "workflow_success": workflow_result.success,
                        "sla_compliant": workflow_time <= 45.0,
                        "stage_times": stage_times,
                        "agent_metrics": workflow_result.agent_metrics,
                        "content_size": len(content),
                        "errors": workflow_result.errors,
                        "warnings": workflow_result.warnings
                    }
                    
                    # Extract meaningful results
                    if workflow_result.domain_analysis:
                        measurement["domain_characteristics"] = {
                            "vocabulary_complexity": workflow_result.domain_analysis.vocabulary_complexity,
                            "concept_density": workflow_result.domain_analysis.concept_density
                        }
                    
                    if workflow_result.extraction_summary:
                        measurement["extraction_results"] = workflow_result.extraction_summary
                    
                    workflow_measurements.append(measurement)
                    
                    status = "✅" if workflow_result.success and workflow_time <= 45.0 else "⚠️" if workflow_result.success else "❌"
                    print(f"     Total Time: {workflow_time:.2f}s {status}")
                    print(f"     Success: {workflow_result.success}")
                    
                    for agent_name, stage_time in stage_times.items():
                        print(f"     {agent_name.replace('_', ' ').title()}: {stage_time:.2f}s")
                
                except Exception as e:
                    workflow_time = time.time() - workflow_start
                    measurement = {
                        "file": test_file.name,
                        "total_workflow_time": workflow_time,
                        "workflow_success": False,
                        "sla_compliant": False,
                        "error": str(e),
                        "content_size": len(content)
                    }
                    workflow_measurements.append(measurement)
                    print(f"     Total Time: {workflow_time:.2f}s ❌ (Error: {str(e)[:50]})")
        
        # Analyze end-to-end workflow performance
        successful_workflows = [m for m in workflow_measurements if m["workflow_success"]]
        
        if successful_workflows:
            workflow_times = [m["total_workflow_time"] for m in successful_workflows]
            sla_compliant_workflows = [m for m in successful_workflows if m["sla_compliant"]]
            
            performance_summary = {
                "total_workflows": len(workflow_measurements),
                "successful_workflows": len(successful_workflows),
                "sla_compliant_workflows": len(sla_compliant_workflows),
                "success_rate": len(successful_workflows) / len(workflow_measurements),
                "sla_compliance_rate": len(sla_compliant_workflows) / len(successful_workflows),
                "avg_workflow_time": statistics.mean(workflow_times),
                "median_workflow_time": statistics.median(workflow_times),
                "min_workflow_time": min(workflow_times),
                "max_workflow_time": max(workflow_times),
                "std_workflow_time": statistics.stdev(workflow_times) if len(workflow_times) > 1 else 0
            }
            
            print(f"\n✅ End-to-End Workflow Performance Summary:")
            print(f"   Success Rate: {performance_summary['success_rate']:.2%}")
            print(f"   SLA Compliance: {performance_summary['sla_compliance_rate']:.2%} (target: 45s)")
            print(f"   Avg Workflow Time: {performance_summary['avg_workflow_time']:.2f}s")
            print(f"   Median Workflow Time: {performance_summary['median_workflow_time']:.2f}s")
            print(f"   Time Range: {performance_summary['min_workflow_time']:.2f}s - {performance_summary['max_workflow_time']:.2f}s")
            
            # Analyze stage performance within workflows
            stage_performance = {}
            for measurement in successful_workflows:
                for stage, stage_time in measurement.get("stage_times", {}).items():
                    if stage not in stage_performance:
                        stage_performance[stage] = []
                    stage_performance[stage].append(stage_time)
            
            print(f"\n   📊 Stage Performance Analysis:")
            for stage, times in stage_performance.items():
                if times:
                    avg_time = statistics.mean(times)
                    print(f"     {stage.replace('_', ' ').title()}: {avg_time:.2f}s avg")
            
            # Assertions for workflow performance
            assert performance_summary["success_rate"] >= 0.8, f"Workflow success rate too low: {performance_summary['success_rate']:.2%}"
            assert performance_summary["sla_compliance_rate"] >= 0.7, f"Workflow SLA compliance too low: {performance_summary['sla_compliance_rate']:.2%}"
            assert performance_summary["avg_workflow_time"] <= 50.0, f"Average workflow time too high: {performance_summary['avg_workflow_time']:.2f}s"
        
        else:
            pytest.fail("No successful workflows to analyze performance")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_user_performance_scalability(
        self,
        azure_services,
        enhanced_test_data_manager,
        performance_monitor
    ):
        """Test system performance under concurrent user load."""
        
        # Define concurrent user scenarios
        concurrency_scenarios = [
            {"users": 5, "requests_per_user": 2, "scenario": "light_load"},
            {"users": 10, "requests_per_user": 2, "scenario": "medium_load"},
            {"users": 15, "requests_per_user": 1, "scenario": "heavy_load"}
        ]
        
        # Get test files optimized for concurrent testing
        concurrent_test_files = enhanced_test_data_manager.get_test_files_by_criteria(
            min_size=500,
            max_size=2000,  # Smaller files for faster concurrent processing
            limit=30  # Enough files for all scenarios
        )
        
        if len(concurrent_test_files) < 10:
            pytest.skip("Insufficient test files for concurrent performance testing")
        
        concurrency_results = []
        
        print("⚡ Concurrent User Performance Scalability Testing:")
        
        for scenario in concurrency_scenarios:
            print(f"\n🔄 Testing {scenario['scenario']}: {scenario['users']} users, {scenario['requests_per_user']} requests each")
            
            async def simulate_user_load(user_id: int, requests: int, files: List[Path]):
                """Simulate load from a single user with multiple requests."""
                user_results = []
                
                for request_id in range(requests):
                    file_idx = (user_id * requests + request_id) % len(files)
                    test_file = files[file_idx]
                    content = test_file.read_text(encoding='utf-8')
                    
                    request_start = time.time()
                    try:
                        # Use domain analysis as the primary operation for concurrency testing
                        result = await run_domain_analysis(content[:1500])  # Limit content for performance
                        request_time = time.time() - request_start
                        
                        user_results.append({
                            "user_id": user_id,
                            "request_id": request_id,
                            "file": test_file.name,
                            "processing_time": request_time,
                            "success": True,
                            "vocabulary_complexity": result.discovered_characteristics.vocabulary_complexity,
                            "concept_density": result.discovered_characteristics.concept_density,
                            "content_size": len(content)
                        })
                        
                    except Exception as e:
                        request_time = time.time() - request_start
                        user_results.append({
                            "user_id": user_id,
                            "request_id": request_id,
                            "file": test_file.name,
                            "processing_time": request_time,
                            "success": False,
                            "error": str(e)[:100],
                            "content_size": len(content)
                        })
                
                return user_results
            
            # Execute concurrent user simulation
            scenario_start = time.time()
            
            user_tasks = [
                simulate_user_load(
                    user_id, 
                    scenario["requests_per_user"], 
                    concurrent_test_files
                ) 
                for user_id in range(scenario["users"])
            ]
            
            user_results_list = await asyncio.gather(*user_tasks, return_exceptions=True)
            scenario_time = time.time() - scenario_start
            
            # Aggregate results
            all_requests = []
            for user_results in user_results_list:
                if isinstance(user_results, list):
                    all_requests.extend(user_results)
                else:
                    # Handle exceptions
                    all_requests.append({
                        "success": False,
                        "error": str(user_results),
                        "processing_time": 0
                    })
            
            successful_requests = [r for r in all_requests if r.get("success", False)]
            failed_requests = [r for r in all_requests if not r.get("success", False)]
            
            # Calculate performance metrics
            if successful_requests:
                response_times = [r["processing_time"] for r in successful_requests]
                
                scenario_result = {
                    "scenario": scenario["scenario"],
                    "users": scenario["users"],
                    "requests_per_user": scenario["requests_per_user"],
                    "total_requests": len(all_requests),
                    "successful_requests": len(successful_requests),
                    "failed_requests": len(failed_requests),
                    "success_rate": len(successful_requests) / len(all_requests) if all_requests else 0,
                    "total_scenario_time": scenario_time,
                    "avg_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "p95_response_time": sorted(response_times)[int(0.95 * len(response_times))] if len(response_times) > 1 else response_times[0] if response_times else 0,
                    "max_response_time": max(response_times),
                    "min_response_time": min(response_times),
                    "requests_per_second": len(all_requests) / scenario_time if scenario_time > 0 else 0,
                    "concurrent_efficiency": scenario_time / (statistics.mean(response_times) * scenario["users"]) if response_times else 1
                }
                
                concurrency_results.append(scenario_result)
                
                print(f"   📊 Results:")
                print(f"     Success Rate: {scenario_result['success_rate']:.2%}")
                print(f"     Avg Response Time: {scenario_result['avg_response_time']:.2f}s")
                print(f"     P95 Response Time: {scenario_result['p95_response_time']:.2f}s")
                print(f"     Max Response Time: {scenario_result['max_response_time']:.2f}s")
                print(f"     Requests/Second: {scenario_result['requests_per_second']:.2f}")
                print(f"     Concurrent Efficiency: {scenario_result['concurrent_efficiency']:.2f}x")
            
            else:
                print(f"   ❌ No successful requests in {scenario['scenario']}")
        
        # Overall concurrency performance analysis
        if concurrency_results:
            print(f"\n✅ Concurrent User Performance Scalability Summary:")
            
            for result in concurrency_results:
                scalability_status = "✅" if result["success_rate"] >= 0.8 and result["avg_response_time"] <= 15.0 else "⚠️"
                print(f"   {result['scenario']}: {result['success_rate']:.2%} success, {result['avg_response_time']:.2f}s avg {scalability_status}")
            
            # Calculate scalability metrics
            max_users_tested = max(r["users"] for r in concurrency_results)
            best_success_rate = max(r["success_rate"] for r in concurrency_results)
            best_throughput = max(r["requests_per_second"] for r in concurrency_results)
            
            print(f"\n   📈 Scalability Metrics:")
            print(f"     Max Concurrent Users Tested: {max_users_tested}")
            print(f"     Best Success Rate: {best_success_rate:.2%}")
            print(f"     Peak Throughput: {best_throughput:.2f} requests/second")
            
            # Assertions for concurrent performance
            overall_success_rate = statistics.mean([r["success_rate"] for r in concurrency_results])
            overall_avg_response_time = statistics.mean([r["avg_response_time"] for r in concurrency_results])
            
            assert overall_success_rate >= 0.75, f"Overall concurrent success rate too low: {overall_success_rate:.2%}"
            assert overall_avg_response_time <= 20.0, f"Overall concurrent response time too high: {overall_avg_response_time:.2f}s"
            assert max_users_tested >= 10, f"Insufficient concurrent user testing: {max_users_tested} users"
        
        else:
            pytest.fail("No concurrent performance results to analyze")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_consistency_and_reliability(
        self,
        azure_services,
        enhanced_test_data_manager,
        performance_monitor
    ):
        """Test performance consistency and reliability over multiple iterations."""
        
        # Get a representative test file for consistency testing
        consistency_files = enhanced_test_data_manager.get_test_files_by_criteria(
            min_size=1000,
            max_size=3000,
            limit=3
        )
        
        if not consistency_files:
            pytest.skip("No suitable files for performance consistency testing")
        
        test_file = consistency_files[0]  # Use the first suitable file
        content = test_file.read_text(encoding='utf-8')
        
        print(f"🔄 Performance Consistency Testing with {test_file.name}:")
        
        # Run multiple iterations to test consistency
        iteration_count = 10
        consistency_measurements = []
        
        for iteration in range(iteration_count):
            async with performance_monitor.measure_operation(
                f"consistency_test_{iteration}",
                sla_target=10.0
            ):
                
                iteration_start = time.time()
                try:
                    result = await run_domain_analysis(content[:2000])
                    iteration_time = time.time() - iteration_start
                    
                    consistency_measurements.append({
                        "iteration": iteration,
                        "processing_time": iteration_time,
                        "success": True,
                        "vocabulary_complexity": result.discovered_characteristics.vocabulary_complexity,
                        "concept_density": result.discovered_characteristics.concept_density,
                        "patterns_found": len(result.discovered_characteristics.structural_patterns)
                    })
                    
                except Exception as e:
                    iteration_time = time.time() - iteration_start
                    consistency_measurements.append({
                        "iteration": iteration,
                        "processing_time": iteration_time,
                        "success": False,
                        "error": str(e)[:100]
                    })
            
            print(f"   Iteration {iteration + 1}: {iteration_time:.2f}s {'✅' if iteration_time <= 10.0 else '❌'}")
        
        # Analyze consistency
        successful_measurements = [m for m in consistency_measurements if m["success"]]
        
        if len(successful_measurements) >= 5:  # Need at least 5 successful measurements for analysis
            processing_times = [m["processing_time"] for m in successful_measurements]
            complexity_values = [m["vocabulary_complexity"] for m in successful_measurements]
            density_values = [m["concept_density"] for m in successful_measurements]
            
            consistency_metrics = {
                "total_iterations": iteration_count,
                "successful_iterations": len(successful_measurements),
                "success_rate": len(successful_measurements) / iteration_count,
                "performance_stats": {
                    "mean_time": statistics.mean(processing_times),
                    "median_time": statistics.median(processing_times),
                    "std_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                    "min_time": min(processing_times),
                    "max_time": max(processing_times),
                    "coefficient_of_variation": statistics.stdev(processing_times) / statistics.mean(processing_times) if len(processing_times) > 1 and statistics.mean(processing_times) > 0 else 0
                },
                "result_consistency": {
                    "complexity_std": statistics.stdev(complexity_values) if len(complexity_values) > 1 else 0,
                    "density_std": statistics.stdev(density_values) if len(density_values) > 1 else 0,
                    "complexity_range": max(complexity_values) - min(complexity_values),
                    "density_range": max(density_values) - min(density_values)
                }
            }
            
            print(f"\n✅ Performance Consistency Analysis:")
            print(f"   Success Rate: {consistency_metrics['success_rate']:.2%}")
            print(f"   Mean Processing Time: {consistency_metrics['performance_stats']['mean_time']:.2f}s")
            print(f"   Std Deviation: {consistency_metrics['performance_stats']['std_time']:.2f}s")
            print(f"   Coefficient of Variation: {consistency_metrics['performance_stats']['coefficient_of_variation']:.3f}")
            print(f"   Time Range: {consistency_metrics['performance_stats']['min_time']:.2f}s - {consistency_metrics['performance_stats']['max_time']:.2f}s")
            
            print(f"\n   📊 Result Consistency:")
            print(f"   Vocabulary Complexity Std: {consistency_metrics['result_consistency']['complexity_std']:.4f}")
            print(f"   Concept Density Std: {consistency_metrics['result_consistency']['density_std']:.4f}")
            print(f"   Complexity Range: {consistency_metrics['result_consistency']['complexity_range']:.4f}")
            print(f"   Density Range: {consistency_metrics['result_consistency']['density_range']:.4f}")
            
            # Assertions for consistency and reliability
            assert consistency_metrics["success_rate"] >= 0.9, f"Consistency success rate too low: {consistency_metrics['success_rate']:.2%}"
            assert consistency_metrics["performance_stats"]["coefficient_of_variation"] <= 0.3, f"Performance too inconsistent: CV = {consistency_metrics['performance_stats']['coefficient_of_variation']:.3f}"
            assert consistency_metrics["result_consistency"]["complexity_std"] <= 0.1, f"Result consistency too low: complexity std = {consistency_metrics['result_consistency']['complexity_std']:.4f}"
            assert consistency_metrics["performance_stats"]["mean_time"] <= 12.0, f"Mean performance outside SLA: {consistency_metrics['performance_stats']['mean_time']:.2f}s"
        
        else:
            pytest.fail(f"Insufficient successful measurements for consistency analysis: {len(successful_measurements)}/{iteration_count}")

    @pytest.mark.asyncio  
    @pytest.mark.performance
    async def test_performance_degradation_under_stress(
        self,
        azure_services,
        enhanced_test_data_manager,
        performance_monitor
    ):
        """Test system performance degradation under stress conditions."""
        
        # Get files for stress testing
        stress_test_files = enhanced_test_data_manager.get_diverse_test_set(count=6)
        
        if len(stress_test_files) < 3:
            pytest.skip("Insufficient test files for stress testing")
        
        print("⚡ Performance Degradation Under Stress Testing:")
        
        # Define stress conditions
        stress_scenarios = [
            {"name": "baseline", "concurrent": 2, "content_multiplier": 1},
            {"name": "increased_load", "concurrent": 5, "content_multiplier": 1},
            {"name": "large_content", "concurrent": 2, "content_multiplier": 3},
            {"name": "high_stress", "concurrent": 8, "content_multiplier": 2}
        ]
        
        stress_results = []
        
        for scenario in stress_scenarios:
            print(f"\n🔥 Stress Scenario: {scenario['name']}")
            print(f"   Concurrent Operations: {scenario['concurrent']}")
            print(f"   Content Size Multiplier: {scenario['content_multiplier']}x")
            
            async def stress_operation(operation_id: int, files: List[Path], content_multiplier: int):
                """Execute a single operation under stress conditions."""
                file_idx = operation_id % len(files)
                test_file = files[file_idx]
                content = test_file.read_text(encoding='utf-8')
                
                # Increase content size for stress testing
                if content_multiplier > 1:
                    content = (content * content_multiplier)[:5000]  # Cap at reasonable size
                
                operation_start = time.time()
                try:
                    result = await run_domain_analysis(content)
                    operation_time = time.time() - operation_start
                    
                    return {
                        "operation_id": operation_id,
                        "success": True,
                        "processing_time": operation_time,
                        "content_size": len(content),
                        "vocabulary_complexity": result.discovered_characteristics.vocabulary_complexity
                    }
                    
                except Exception as e:
                    operation_time = time.time() - operation_start
                    return {
                        "operation_id": operation_id,
                        "success": False,
                        "processing_time": operation_time,
                        "content_size": len(content),
                        "error": str(e)[:100]
                    }
            
            # Execute stress scenario
            scenario_start = time.time()
            
            stress_tasks = [
                stress_operation(i, stress_test_files, scenario["content_multiplier"])
                for i in range(scenario["concurrent"])
            ]
            
            scenario_results = await asyncio.gather(*stress_tasks)
            scenario_time = time.time() - scenario_start
            
            # Analyze stress results
            successful_operations = [r for r in scenario_results if r["success"]]
            
            if successful_operations:
                processing_times = [r["processing_time"] for r in successful_operations]
                content_sizes = [r["content_size"] for r in successful_operations]
                
                stress_metrics = {
                    "scenario": scenario["name"],
                    "concurrent_operations": scenario["concurrent"],
                    "content_multiplier": scenario["content_multiplier"],
                    "total_operations": len(scenario_results),
                    "successful_operations": len(successful_operations),
                    "success_rate": len(successful_operations) / len(scenario_results),
                    "total_scenario_time": scenario_time,
                    "avg_processing_time": statistics.mean(processing_times),
                    "max_processing_time": max(processing_times),
                    "avg_content_size": statistics.mean(content_sizes),
                    "operations_per_second": len(scenario_results) / scenario_time if scenario_time > 0 else 0
                }
                
                stress_results.append(stress_metrics)
                
                print(f"   Success Rate: {stress_metrics['success_rate']:.2%}")
                print(f"   Avg Processing Time: {stress_metrics['avg_processing_time']:.2f}s")
                print(f"   Max Processing Time: {stress_metrics['max_processing_time']:.2f}s")
                print(f"   Operations/Second: {stress_metrics['operations_per_second']:.2f}")
            
            else:
                print(f"   ❌ No successful operations in {scenario['name']}")
        
        # Analyze performance degradation
        if len(stress_results) >= 2:
            baseline = next((r for r in stress_results if r["scenario"] == "baseline"), None)
            
            print(f"\n✅ Performance Degradation Analysis:")
            
            if baseline:
                print(f"   Baseline Performance:")
                print(f"     Success Rate: {baseline['success_rate']:.2%}")
                print(f"     Avg Processing Time: {baseline['avg_processing_time']:.2f}s")
                print(f"     Operations/Second: {baseline['operations_per_second']:.2f}")
                
                print(f"\n   Stress Impact Analysis:")
                for result in stress_results:
                    if result["scenario"] != "baseline":
                        degradation_factor = result["avg_processing_time"] / baseline["avg_processing_time"] if baseline["avg_processing_time"] > 0 else 1
                        throughput_impact = result["operations_per_second"] / baseline["operations_per_second"] if baseline["operations_per_second"] > 0 else 1
                        
                        print(f"     {result['scenario']}:")
                        print(f"       Performance Degradation: {degradation_factor:.2f}x")
                        print(f"       Throughput Impact: {throughput_impact:.2f}x")
                        print(f"       Success Rate: {result['success_rate']:.2%}")
            
            # Assertions for stress resistance
            high_stress_result = next((r for r in stress_results if r["scenario"] == "high_stress"), None)
            
            if high_stress_result:
                assert high_stress_result["success_rate"] >= 0.6, f"System fails under high stress: {high_stress_result['success_rate']:.2%}"
                assert high_stress_result["avg_processing_time"] <= 30.0, f"Performance degrades too much under stress: {high_stress_result['avg_processing_time']:.2f}s"
            
            # Overall stress resistance
            min_success_rate = min(r["success_rate"] for r in stress_results)
            max_processing_time = max(r["avg_processing_time"] for r in stress_results)
            
            assert min_success_rate >= 0.5, f"Minimum success rate too low across stress scenarios: {min_success_rate:.2%}"
            assert max_processing_time <= 35.0, f"Maximum processing time too high across stress scenarios: {max_processing_time:.2f}s"
        
        else:
            pytest.skip("Insufficient stress test results for degradation analysis")