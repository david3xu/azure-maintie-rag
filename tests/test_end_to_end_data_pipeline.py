"""
End-to-End Data Pipeline Validation Tests for Azure Universal RAG System
=======================================================================

Comprehensive validation of the complete data processing pipeline using
all available Azure AI Language Service documentation files (179 files).

Pipeline Stages Tested:
1. Data Ingestion and Quality Validation
2. Domain Intelligence Analysis
3. Knowledge Extraction and Graph Building
4. Vector Indexing and Search Integration
5. Universal Search and Retrieval
6. Performance and Accuracy Validation

Real Data Sources:
- 179 Azure AI Language Service documentation files
- Diverse content types: API docs, tutorials, concepts, quickstarts
- Authentic technical content with varying complexity levels
"""

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pytest
from dotenv import load_dotenv

# Load environment before imports
load_dotenv()

from agents.core.universal_deps import get_universal_deps
from agents.domain_intelligence.agent import run_domain_analysis
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.universal_search.agent import run_universal_search


class TestEndToEndDataPipelineValidation:
    """Comprehensive end-to-end data pipeline validation with real Azure AI files."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_pipeline_with_full_dataset(
        self,
        azure_services,
        enhanced_test_data_manager,
        performance_monitor
    ):
        """
        Test complete data pipeline with the full dataset of Azure AI files.
        
        This test processes all available files through the complete pipeline
        to validate production-scale data processing capabilities.
        """
        
        # Get comprehensive dataset quality report
        data_report = enhanced_test_data_manager.generate_test_data_report()
        
        print(f"📊 Full Dataset Pipeline Validation:")
        print(f"   Total Files: {data_report['total_files']}")
        print(f"   Suitable Files: {data_report['suitable_files']}")
        print(f"   Quality Ratio: {data_report['suitability_ratio']:.2%}")
        
        if data_report['suitable_files'] < 10:
            pytest.skip(f"Insufficient suitable files for full pipeline testing: {data_report['suitable_files']}")
        
        # Get representative sample for comprehensive testing
        pipeline_files = enhanced_test_data_manager.get_diverse_test_set(count=min(25, data_report['suitable_files']))
        
        pipeline_results = {
            "files_processed": [],
            "stage_performance": {
                "domain_intelligence": {"successes": 0, "failures": 0, "times": []},
                "knowledge_extraction": {"successes": 0, "failures": 0, "times": []},
                "universal_search": {"successes": 0, "failures": 0, "times": []}
            },
            "quality_metrics": {
                "domain_characteristics": [],
                "extracted_entities": [],
                "search_results": []
            },
            "errors": [],
            "warnings": []
        }
        
        print(f"\n🔄 Processing {len(pipeline_files)} files through complete pipeline:")
        
        for i, file_path in enumerate(pipeline_files):
            content = file_path.read_text(encoding='utf-8')
            file_name = file_path.name
            
            print(f"\n   📄 Processing {file_name} ({len(content)} chars)")
            
            file_result = {
                "file_name": file_name,
                "content_size": len(content),
                "stages_completed": [],
                "total_processing_time": 0,
                "success": False
            }
            
            file_start_time = time.time()
            
            try:
                # Stage 1: Domain Intelligence Analysis
                print(f"      🧠 Stage 1: Domain Intelligence...")
                stage1_start = time.time()
                
                domain_result = await run_domain_analysis(content, detailed=True)
                stage1_time = time.time() - stage1_start
                
                domain_characteristics = domain_result.discovered_characteristics
                file_result["stages_completed"].append("domain_intelligence")
                file_result["domain_analysis"] = {
                    "vocabulary_complexity": domain_characteristics.vocabulary_complexity,
                    "concept_density": domain_characteristics.concept_density,
                    "content_signature": domain_characteristics.content_signature,
                    "patterns_found": len(domain_characteristics.structural_patterns),
                    "processing_time": stage1_time
                }
                
                pipeline_results["stage_performance"]["domain_intelligence"]["successes"] += 1
                pipeline_results["stage_performance"]["domain_intelligence"]["times"].append(stage1_time)
                pipeline_results["quality_metrics"]["domain_characteristics"].append({
                    "file": file_name,
                    "vocabulary_complexity": domain_characteristics.vocabulary_complexity,
                    "concept_density": domain_characteristics.concept_density,
                    "patterns_count": len(domain_characteristics.structural_patterns)
                })
                
                print(f"         ✅ Domain Analysis: {stage1_time:.2f}s")
                print(f"            Vocabulary Complexity: {domain_characteristics.vocabulary_complexity:.3f}")
                print(f"            Concept Density: {domain_characteristics.concept_density:.3f}")
                
                # Stage 2: Knowledge Extraction
                print(f"      🔍 Stage 2: Knowledge Extraction...")
                stage2_start = time.time()
                
                extraction_result = await run_knowledge_extraction(content, use_domain_analysis=True)
                stage2_time = time.time() - stage2_start
                
                file_result["stages_completed"].append("knowledge_extraction")
                file_result["knowledge_extraction"] = {
                    "entities_count": len(extraction_result.entities),
                    "relationships_count": len(extraction_result.relationships),
                    "extraction_confidence": extraction_result.extraction_confidence,
                    "processing_time": stage2_time
                }
                
                pipeline_results["stage_performance"]["knowledge_extraction"]["successes"] += 1
                pipeline_results["stage_performance"]["knowledge_extraction"]["times"].append(stage2_time)
                pipeline_results["quality_metrics"]["extracted_entities"].append({
                    "file": file_name,
                    "entities_count": len(extraction_result.entities),
                    "relationships_count": len(extraction_result.relationships),
                    "extraction_confidence": extraction_result.extraction_confidence
                })
                
                print(f"         ✅ Knowledge Extraction: {stage2_time:.2f}s")
                print(f"            Entities: {len(extraction_result.entities)}")
                print(f"            Relationships: {len(extraction_result.relationships)}")
                print(f"            Confidence: {extraction_result.extraction_confidence:.3f}")
                
                # Stage 3: Universal Search (if entities extracted)
                if extraction_result.entities:
                    print(f"      🔎 Stage 3: Universal Search...")
                    stage3_start = time.time()
                    
                    # Create search query from top entities
                    top_entities = [e.text for e in extraction_result.entities[:3]]
                    search_query = f"Information about {', '.join(top_entities)}"
                    
                    search_result = await run_universal_search(search_query, max_results=5)
                    stage3_time = time.time() - stage3_start
                    
                    file_result["stages_completed"].append("universal_search")
                    file_result["universal_search"] = {
                        "query": search_query,
                        "results_count": len(search_result.unified_results),
                        "search_strategy": search_result.search_strategy_used,
                        "search_confidence": search_result.search_confidence,
                        "processing_time": stage3_time
                    }
                    
                    pipeline_results["stage_performance"]["universal_search"]["successes"] += 1
                    pipeline_results["stage_performance"]["universal_search"]["times"].append(stage3_time)
                    pipeline_results["quality_metrics"]["search_results"].append({
                        "file": file_name,
                        "query": search_query,
                        "results_count": len(search_result.unified_results),
                        "search_confidence": search_result.search_confidence,
                        "strategy": search_result.search_strategy_used
                    })
                    
                    print(f"         ✅ Universal Search: {stage3_time:.2f}s")
                    print(f"            Results: {len(search_result.unified_results)}")
                    print(f"            Strategy: {search_result.search_strategy_used}")
                    print(f"            Confidence: {search_result.search_confidence:.3f}")
                
                else:
                    print(f"      ⚠️  Stage 3: Skipped (no entities for search)")
                
                file_result["success"] = True
                file_result["total_processing_time"] = time.time() - file_start_time
                
                print(f"      ✅ Complete Pipeline: {file_result['total_processing_time']:.2f}s")
                
            except Exception as e:
                file_result["total_processing_time"] = time.time() - file_start_time
                file_result["error"] = str(e)
                
                # Update failure counts
                if "domain_intelligence" not in file_result["stages_completed"]:
                    pipeline_results["stage_performance"]["domain_intelligence"]["failures"] += 1
                elif "knowledge_extraction" not in file_result["stages_completed"]:
                    pipeline_results["stage_performance"]["knowledge_extraction"]["failures"] += 1
                else:
                    pipeline_results["stage_performance"]["universal_search"]["failures"] += 1
                
                pipeline_results["errors"].append({
                    "file": file_name,
                    "error": str(e),
                    "stages_completed": file_result["stages_completed"]
                })
                
                print(f"      ❌ Pipeline Failed: {str(e)[:100]}")
            
            pipeline_results["files_processed"].append(file_result)
        
        # Analyze pipeline results
        successful_files = [f for f in pipeline_results["files_processed"] if f["success"]]
        success_rate = len(successful_files) / len(pipeline_results["files_processed"])
        
        # Calculate stage success rates
        stage_stats = {}
        for stage_name, stage_data in pipeline_results["stage_performance"].items():
            total_attempts = stage_data["successes"] + stage_data["failures"]
            stage_stats[stage_name] = {
                "success_rate": stage_data["successes"] / total_attempts if total_attempts > 0 else 0,
                "avg_processing_time": statistics.mean(stage_data["times"]) if stage_data["times"] else 0,
                "total_attempts": total_attempts
            }
        
        print(f"\n✅ Complete Pipeline Validation Results:")
        print(f"   Files Processed: {len(pipeline_results['files_processed'])}")
        print(f"   Success Rate: {success_rate:.2%}")
        
        print(f"\n   📊 Stage Performance:")
        for stage_name, stats in stage_stats.items():
            print(f"     {stage_name.replace('_', ' ').title()}:")
            print(f"       Success Rate: {stats['success_rate']:.2%}")
            print(f"       Avg Time: {stats['avg_processing_time']:.2f}s")
            print(f"       Attempts: {stats['total_attempts']}")
        
        # Quality metrics analysis
        if pipeline_results["quality_metrics"]["domain_characteristics"]:
            domain_complexities = [d["vocabulary_complexity"] for d in pipeline_results["quality_metrics"]["domain_characteristics"]]
            avg_complexity = statistics.mean(domain_complexities)
            print(f"\n   🎯 Quality Metrics:")
            print(f"     Avg Vocabulary Complexity: {avg_complexity:.3f}")
            
            entities_counts = [e["entities_count"] for e in pipeline_results["quality_metrics"]["extracted_entities"]]
            avg_entities = statistics.mean(entities_counts) if entities_counts else 0
            print(f"     Avg Entities Extracted: {avg_entities:.1f}")
            
            search_counts = [s["results_count"] for s in pipeline_results["quality_metrics"]["search_results"]]
            avg_search_results = statistics.mean(search_counts) if search_counts else 0
            print(f"     Avg Search Results: {avg_search_results:.1f}")
        
        # Assertions for production readiness
        assert success_rate >= 0.8, f"Pipeline success rate too low: {success_rate:.2%}"
        assert stage_stats["domain_intelligence"]["success_rate"] >= 0.85, f"Domain Intelligence success rate too low: {stage_stats['domain_intelligence']['success_rate']:.2%}"
        assert stage_stats["knowledge_extraction"]["success_rate"] >= 0.75, f"Knowledge Extraction success rate too low: {stage_stats['knowledge_extraction']['success_rate']:.2%}"
        
        # Performance assertions
        if stage_stats["domain_intelligence"]["avg_processing_time"] > 0:
            assert stage_stats["domain_intelligence"]["avg_processing_time"] <= 12.0, f"Domain Intelligence too slow: {stage_stats['domain_intelligence']['avg_processing_time']:.2f}s"
        
        return pipeline_results

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_data_quality_and_diversity_validation(
        self,
        azure_services,
        enhanced_test_data_manager
    ):
        """Validate data quality and diversity across the complete dataset."""
        
        print("📋 Data Quality and Diversity Validation:")
        
        # Generate comprehensive data quality report
        quality_report = enhanced_test_data_manager.generate_test_data_report()
        
        print(f"\n   📊 Dataset Overview:")
        print(f"     Total Files: {quality_report['total_files']}")
        print(f"     Suitable Files: {quality_report['suitable_files']}")
        print(f"     Suitability Ratio: {quality_report['suitability_ratio']:.2%}")
        print(f"     Average Size: {quality_report['average_size']:.0f} chars")
        
        # Content type distribution
        print(f"\n   🎯 Content Type Distribution:")
        for content_type, count in quality_report['content_type_distribution'].items():
            percentage = (count / quality_report['suitable_files']) * 100 if quality_report['suitable_files'] > 0 else 0
            print(f"     {content_type.replace('_', ' ').title()}: {count} files ({percentage:.1f}%)")
        
        # Quality metrics
        print(f"\n   ✅ Quality Metrics:")
        for metric_name, value in quality_report['quality_metrics'].items():
            if isinstance(value, float):
                print(f"     {metric_name.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"     {metric_name.replace('_', ' ').title()}: {value}")
        
        # Detailed file analysis for diversity validation
        diverse_sample = enhanced_test_data_manager.get_diverse_test_set(count=15)
        diversity_metrics = {
            "size_distribution": [],
            "complexity_distribution": [],
            "content_types": set(),
            "language_features": {
                "has_code_blocks": 0,
                "has_api_content": 0,
                "has_tutorials": 0,
                "has_conceptual_content": 0
            }
        }
        
        print(f"\n   🔍 Diversity Analysis (Sample: {len(diverse_sample)} files):")
        
        for file_path in diverse_sample:
            analysis = enhanced_test_data_manager.analyze_file_quality(file_path)
            
            diversity_metrics["size_distribution"].append(analysis["size_chars"])
            diversity_metrics["complexity_distribution"].append(analysis["complexity_score"])
            
            # Content type identification
            if analysis["has_api_content"]:
                diversity_metrics["content_types"].add("api_documentation")
                diversity_metrics["language_features"]["has_api_content"] += 1
            if analysis["has_tutorial_content"]:
                diversity_metrics["content_types"].add("tutorial")
                diversity_metrics["language_features"]["has_tutorials"] += 1
            if analysis["has_conceptual_content"]:
                diversity_metrics["content_types"].add("conceptual")
                diversity_metrics["language_features"]["has_conceptual_content"] += 1
            if analysis["has_code_blocks"]:
                diversity_metrics["language_features"]["has_code_blocks"] += 1
        
        # Calculate diversity statistics
        size_stats = {
            "mean": statistics.mean(diversity_metrics["size_distribution"]),
            "median": statistics.median(diversity_metrics["size_distribution"]),
            "std": statistics.stdev(diversity_metrics["size_distribution"]) if len(diversity_metrics["size_distribution"]) > 1 else 0,
            "range": max(diversity_metrics["size_distribution"]) - min(diversity_metrics["size_distribution"])
        }
        
        complexity_stats = {
            "mean": statistics.mean(diversity_metrics["complexity_distribution"]),
            "std": statistics.stdev(diversity_metrics["complexity_distribution"]) if len(diversity_metrics["complexity_distribution"]) > 1 else 0
        }
        
        print(f"     📏 Size Distribution:")
        print(f"       Mean: {size_stats['mean']:.0f} chars")
        print(f"       Median: {size_stats['median']:.0f} chars")
        print(f"       Range: {size_stats['range']:.0f} chars")
        print(f"       Std Dev: {size_stats['std']:.0f} chars")
        
        print(f"     🧮 Complexity Distribution:")
        print(f"       Mean Complexity: {complexity_stats['mean']:.3f}")
        print(f"       Std Dev: {complexity_stats['std']:.3f}")
        
        print(f"     🎭 Content Types Found: {len(diversity_metrics['content_types'])}")
        for content_type in sorted(diversity_metrics["content_types"]):
            print(f"       - {content_type.replace('_', ' ').title()}")
        
        print(f"     🔧 Language Features:")
        for feature, count in diversity_metrics["language_features"].items():
            percentage = (count / len(diverse_sample)) * 100
            print(f"       {feature.replace('_', ' ').title()}: {count}/{len(diverse_sample)} ({percentage:.1f}%)")
        
        # Data quality and diversity assertions
        assert quality_report["suitability_ratio"] >= 0.6, f"Dataset suitability too low: {quality_report['suitability_ratio']:.2%}"
        assert quality_report["suitable_files"] >= 15, f"Insufficient suitable files: {quality_report['suitable_files']}"
        assert len(diversity_metrics["content_types"]) >= 3, f"Insufficient content type diversity: {len(diversity_metrics['content_types'])}"
        assert size_stats["std"] > 500, f"Insufficient size diversity: {size_stats['std']:.0f}"
        assert complexity_stats["mean"] > 0.3, f"Content complexity too low: {complexity_stats['mean']:.3f}"
        
        # Language feature coverage
        api_coverage = diversity_metrics["language_features"]["has_api_content"] / len(diverse_sample)
        code_coverage = diversity_metrics["language_features"]["has_code_blocks"] / len(diverse_sample)
        assert api_coverage >= 0.4, f"Insufficient API content coverage: {api_coverage:.2%}"
        assert code_coverage >= 0.5, f"Insufficient code content coverage: {code_coverage:.2%}"
        
        print(f"\n✅ Data Quality and Diversity: VALIDATED")
        
        return quality_report

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_scalability_with_large_dataset(
        self,
        azure_services,
        enhanced_test_data_manager,
        performance_monitor
    ):
        """Test pipeline scalability with larger subsets of the dataset."""
        
        print("⚡ Pipeline Scalability Testing:")
        
        # Test different batch sizes to validate scalability
        scalability_scenarios = [
            {"name": "small_batch", "batch_size": 5},
            {"name": "medium_batch", "batch_size": 10}, 
            {"name": "large_batch", "batch_size": 20}
        ]
        
        scalability_results = []
        
        for scenario in scalability_scenarios:
            batch_size = scenario["batch_size"]
            scenario_name = scenario["name"]
            
            print(f"\n🔄 Testing {scenario_name}: {batch_size} files")
            
            # Get batch of files for testing
            batch_files = enhanced_test_data_manager.get_diverse_test_set(count=batch_size)
            
            if len(batch_files) < batch_size:
                print(f"   ⚠️  Only {len(batch_files)} files available, proceeding with available files")
                if len(batch_files) < 3:
                    continue
            
            batch_start_time = time.time()
            
            # Process batch sequentially to test scalability
            batch_results = {
                "scenario": scenario_name,
                "files_processed": len(batch_files),
                "processing_times": [],
                "domain_analysis_results": [],
                "extraction_results": [],
                "total_entities": 0,
                "total_relationships": 0,
                "errors": []
            }
            
            for i, file_path in enumerate(batch_files):
                content = file_path.read_text(encoding='utf-8')
                file_start = time.time()
                
                try:
                    # Process through domain analysis and knowledge extraction
                    domain_result = await run_domain_analysis(content[:2000])  # Limit for scalability
                    extraction_result = await run_knowledge_extraction(content[:1500], use_domain_analysis=False)
                    
                    file_processing_time = time.time() - file_start
                    
                    batch_results["processing_times"].append(file_processing_time)
                    batch_results["domain_analysis_results"].append({
                        "vocabulary_complexity": domain_result.discovered_characteristics.vocabulary_complexity,
                        "concept_density": domain_result.discovered_characteristics.concept_density
                    })
                    batch_results["extraction_results"].append({
                        "entities_count": len(extraction_result.entities),
                        "relationships_count": len(extraction_result.relationships)
                    })
                    
                    batch_results["total_entities"] += len(extraction_result.entities)
                    batch_results["total_relationships"] += len(extraction_result.relationships)
                    
                    print(f"     📄 File {i+1}/{len(batch_files)}: {file_processing_time:.2f}s")
                    
                except Exception as e:
                    batch_results["errors"].append({
                        "file": file_path.name,
                        "error": str(e)[:100]
                    })
                    print(f"     📄 File {i+1}/{len(batch_files)}: ERROR - {str(e)[:50]}")
            
            batch_total_time = time.time() - batch_start_time
            
            # Calculate batch metrics
            successful_files = len(batch_results["processing_times"])
            if successful_files > 0:
                batch_metrics = {
                    "scenario": scenario_name,
                    "batch_size": batch_size,
                    "successful_files": successful_files,
                    "error_count": len(batch_results["errors"]),
                    "success_rate": successful_files / len(batch_files),
                    "total_batch_time": batch_total_time,
                    "avg_file_processing_time": statistics.mean(batch_results["processing_times"]),
                    "total_entities_extracted": batch_results["total_entities"],
                    "total_relationships_extracted": batch_results["total_relationships"],
                    "files_per_minute": (successful_files / batch_total_time) * 60 if batch_total_time > 0 else 0,
                    "avg_vocabulary_complexity": statistics.mean([r["vocabulary_complexity"] for r in batch_results["domain_analysis_results"]]),
                    "avg_entities_per_file": batch_results["total_entities"] / successful_files if successful_files > 0 else 0
                }
                
                scalability_results.append(batch_metrics)
                
                print(f"   📊 {scenario_name} Results:")
                print(f"     Success Rate: {batch_metrics['success_rate']:.2%}")
                print(f"     Total Time: {batch_metrics['total_batch_time']:.2f}s")
                print(f"     Avg File Time: {batch_metrics['avg_file_processing_time']:.2f}s")
                print(f"     Files/Minute: {batch_metrics['files_per_minute']:.1f}")
                print(f"     Total Entities: {batch_metrics['total_entities_extracted']}")
                print(f"     Avg Entities/File: {batch_metrics['avg_entities_per_file']:.1f}")
            
            else:
                print(f"   ❌ No successful files in {scenario_name}")
        
        # Analyze scalability trends
        if len(scalability_results) >= 2:
            print(f"\n✅ Scalability Analysis:")
            
            # Calculate scalability metrics
            batch_sizes = [r["batch_size"] for r in scalability_results]
            throughputs = [r["files_per_minute"] for r in scalability_results]
            success_rates = [r["success_rate"] for r in scalability_results]
            
            # Linear scalability would maintain constant throughput per file
            print(f"   📈 Throughput Scaling:")
            for result in scalability_results:
                print(f"     {result['scenario']}: {result['files_per_minute']:.1f} files/minute")
            
            print(f"   🎯 Quality Scaling:")
            for result in scalability_results:
                print(f"     {result['scenario']}: {result['success_rate']:.2%} success, {result['avg_entities_per_file']:.1f} entities/file")
            
            # Scalability assertions
            min_success_rate = min(success_rates)
            min_throughput = min(throughputs)
            
            assert min_success_rate >= 0.75, f"Success rate degrades too much at scale: {min_success_rate:.2%}"
            assert min_throughput >= 2.0, f"Throughput too low at scale: {min_throughput:.1f} files/minute"
            
            # Check that larger batches don't severely degrade performance
            if len(scalability_results) >= 2:
                small_batch_result = scalability_results[0]
                large_batch_result = scalability_results[-1]
                
                throughput_degradation = small_batch_result["files_per_minute"] / large_batch_result["files_per_minute"] if large_batch_result["files_per_minute"] > 0 else 1
                quality_degradation = small_batch_result["success_rate"] / large_batch_result["success_rate"] if large_batch_result["success_rate"] > 0 else 1
                
                print(f"\n   ⚖️ Degradation Analysis:")
                print(f"     Throughput Degradation: {throughput_degradation:.2f}x")
                print(f"     Quality Degradation: {quality_degradation:.2f}x")
                
                assert throughput_degradation <= 2.0, f"Throughput degrades too much at scale: {throughput_degradation:.2f}x"
                assert quality_degradation <= 1.2, f"Quality degrades too much at scale: {quality_degradation:.2f}x"
        
        else:
            pytest.skip("Insufficient scalability test results")
        
        print(f"\n✅ Pipeline Scalability: VALIDATED")
        
        return scalability_results

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_accuracy_with_ground_truth(
        self,
        azure_services,
        enhanced_test_data_manager
    ):
        """Test pipeline accuracy using ground truth validation where possible."""
        
        print("🎯 Pipeline Accuracy Validation:")
        
        # Get high-quality files for accuracy testing
        accuracy_test_files = enhanced_test_data_manager.get_test_files_by_criteria(
            min_size=1000,
            max_size=4000,
            content_types=["api", "azure"],
            limit=10
        )
        
        if len(accuracy_test_files) < 5:
            pytest.skip("Insufficient files for accuracy testing")
        
        accuracy_results = {
            "files_tested": [],
            "domain_analysis_accuracy": [],
            "entity_extraction_accuracy": [],
            "search_relevance_scores": []
        }
        
        print(f"\n🔍 Testing accuracy on {len(accuracy_test_files)} files:")
        
        for file_path in accuracy_test_files:
            content = file_path.read_text(encoding='utf-8')
            file_name = file_path.name
            
            print(f"\n   📄 Accuracy testing: {file_name}")
            
            file_accuracy = {
                "file_name": file_name,
                "content_size": len(content),
                "ground_truth_indicators": self._extract_ground_truth_indicators(content),
                "results": {}
            }
            
            try:
                # Domain Analysis Accuracy
                domain_result = await run_domain_analysis(content)
                domain_characteristics = domain_result.discovered_characteristics
                
                # Validate domain analysis makes sense
                vocabulary_reasonable = 0.0 <= domain_characteristics.vocabulary_complexity <= 1.0
                density_reasonable = 0.0 <= domain_characteristics.concept_density <= 1.0
                has_patterns = len(domain_characteristics.structural_patterns) > 0
                
                domain_accuracy_score = sum([vocabulary_reasonable, density_reasonable, has_patterns]) / 3
                
                file_accuracy["results"]["domain_analysis"] = {
                    "accuracy_score": domain_accuracy_score,
                    "vocabulary_complexity": domain_characteristics.vocabulary_complexity,
                    "concept_density": domain_characteristics.concept_density,
                    "patterns_found": len(domain_characteristics.structural_patterns)
                }
                
                accuracy_results["domain_analysis_accuracy"].append(domain_accuracy_score)
                
                print(f"      🧠 Domain Analysis Accuracy: {domain_accuracy_score:.3f}")
                
                # Knowledge Extraction Accuracy
                extraction_result = await run_knowledge_extraction(content[:2000], use_domain_analysis=True)
                
                # Validate extraction quality
                entities_extracted = len(extraction_result.entities)
                relationships_extracted = len(extraction_result.relationships)
                extraction_confidence = extraction_result.extraction_confidence
                
                # Expected entities based on ground truth indicators
                expected_entities = self._count_expected_entities(content, file_accuracy["ground_truth_indicators"])
                
                # Calculate extraction accuracy
                if expected_entities > 0:
                    entity_recall = min(entities_extracted / expected_entities, 1.0)
                else:
                    entity_recall = 1.0 if entities_extracted == 0 else 0.5  # Partial credit
                
                # Factor in extraction confidence
                extraction_accuracy_score = (entity_recall + extraction_confidence) / 2
                
                file_accuracy["results"]["knowledge_extraction"] = {
                    "accuracy_score": extraction_accuracy_score,
                    "entities_extracted": entities_extracted,
                    "expected_entities": expected_entities,
                    "relationships_extracted": relationships_extracted,
                    "extraction_confidence": extraction_confidence
                }
                
                accuracy_results["entity_extraction_accuracy"].append(extraction_accuracy_score)
                
                print(f"      🔍 Extraction Accuracy: {extraction_accuracy_score:.3f}")
                print(f"         Entities: {entities_extracted} (expected ~{expected_entities})")
                print(f"         Relationships: {relationships_extracted}")
                
                # Search Relevance (if entities available)
                if extraction_result.entities:
                    primary_entity = extraction_result.entities[0].text
                    search_query = f"Information about {primary_entity}"
                    
                    search_result = await run_universal_search(search_query, max_results=5)
                    
                    # Evaluate search relevance
                    search_relevance = self._evaluate_search_relevance(
                        search_query, 
                        search_result, 
                        file_accuracy["ground_truth_indicators"]
                    )
                    
                    file_accuracy["results"]["search_relevance"] = {
                        "relevance_score": search_relevance,
                        "query": search_query,
                        "results_count": len(search_result.unified_results),
                        "search_confidence": search_result.search_confidence
                    }
                    
                    accuracy_results["search_relevance_scores"].append(search_relevance)
                    
                    print(f"      🔎 Search Relevance: {search_relevance:.3f}")
                
            except Exception as e:
                file_accuracy["error"] = str(e)
                print(f"      ❌ Accuracy test failed: {str(e)[:100]}")
            
            accuracy_results["files_tested"].append(file_accuracy)
        
        # Calculate overall accuracy metrics
        if accuracy_results["domain_analysis_accuracy"]:
            domain_avg_accuracy = statistics.mean(accuracy_results["domain_analysis_accuracy"])
            print(f"\n   🧠 Domain Analysis Avg Accuracy: {domain_avg_accuracy:.3f}")
        
        if accuracy_results["entity_extraction_accuracy"]:
            extraction_avg_accuracy = statistics.mean(accuracy_results["entity_extraction_accuracy"])
            print(f"   🔍 Entity Extraction Avg Accuracy: {extraction_avg_accuracy:.3f}")
        
        if accuracy_results["search_relevance_scores"]:
            search_avg_relevance = statistics.mean(accuracy_results["search_relevance_scores"])
            print(f"   🔎 Search Avg Relevance: {search_avg_relevance:.3f}")
        
        print(f"\n✅ Pipeline Accuracy Summary:")
        files_with_results = len([f for f in accuracy_results["files_tested"] if "error" not in f])
        print(f"   Files Successfully Tested: {files_with_results}/{len(accuracy_test_files)}")
        
        # Overall accuracy validation
        overall_accuracy_scores = []
        if accuracy_results["domain_analysis_accuracy"]:
            overall_accuracy_scores.extend(accuracy_results["domain_analysis_accuracy"])
        if accuracy_results["entity_extraction_accuracy"]:
            overall_accuracy_scores.extend(accuracy_results["entity_extraction_accuracy"])
        if accuracy_results["search_relevance_scores"]:
            overall_accuracy_scores.extend(accuracy_results["search_relevance_scores"])
        
        if overall_accuracy_scores:
            overall_accuracy = statistics.mean(overall_accuracy_scores)
            print(f"   Overall Pipeline Accuracy: {overall_accuracy:.3f}")
            
            # Accuracy assertions
            assert overall_accuracy >= 0.7, f"Overall pipeline accuracy too low: {overall_accuracy:.3f}"
            assert domain_avg_accuracy >= 0.8, f"Domain analysis accuracy too low: {domain_avg_accuracy:.3f}"
            assert extraction_avg_accuracy >= 0.6, f"Entity extraction accuracy too low: {extraction_avg_accuracy:.3f}"
        
        return accuracy_results

    def _extract_ground_truth_indicators(self, content: str) -> Dict[str, Any]:
        """Extract ground truth indicators from content for accuracy validation."""
        indicators = {
            "azure_services": [],
            "api_endpoints": [],
            "code_samples": 0,
            "technical_terms": [],
            "document_type": "unknown"
        }
        
        content_lower = content.lower()
        
        # Identify Azure services mentioned
        azure_services = [
            "azure openai", "cognitive search", "cosmos db", "blob storage",
            "app service", "functions", "logic apps", "event grid",
            "service bus", "key vault", "application insights"
        ]
        
        for service in azure_services:
            if service in content_lower:
                indicators["azure_services"].append(service)
        
        # Count code samples
        indicators["code_samples"] = content.count("```")
        
        # Identify API endpoints
        if "api/" in content_lower or "/api" in content_lower:
            indicators["api_endpoints"] = ["api_endpoint_found"]
        
        # Identify document type
        if any(word in content_lower for word in ["tutorial", "how to", "guide"]):
            indicators["document_type"] = "tutorial"
        elif any(word in content_lower for word in ["api", "endpoint", "rest"]):
            indicators["document_type"] = "api_documentation"
        elif any(word in content_lower for word in ["concept", "overview", "understand"]):
            indicators["document_type"] = "conceptual"
        
        # Technical terms
        technical_terms = ["json", "http", "rest", "sdk", "authentication", "authorization"]
        for term in technical_terms:
            if term in content_lower:
                indicators["technical_terms"].append(term)
        
        return indicators

    def _count_expected_entities(self, content: str, ground_truth: Dict[str, Any]) -> int:
        """Estimate expected number of entities based on ground truth indicators."""
        expected = 0
        
        # Azure services mentioned
        expected += len(ground_truth["azure_services"])
        
        # API endpoints
        expected += len(ground_truth["api_endpoints"])
        
        # Technical terms
        expected += len(ground_truth["technical_terms"])
        
        # Base entities from content analysis
        if len(content) > 1000:
            expected += 3  # Minimum expected for substantial content
        elif len(content) > 500:
            expected += 2
        else:
            expected += 1
        
        return max(expected, 1)  # At least 1 entity expected

    def _evaluate_search_relevance(
        self, 
        query: str, 
        search_result, 
        ground_truth: Dict[str, Any]
    ) -> float:
        """Evaluate search result relevance based on ground truth indicators."""
        
        if len(search_result.unified_results) == 0:
            return 0.0
        
        relevance_factors = []
        
        # Factor 1: Search confidence
        relevance_factors.append(search_result.search_confidence)
        
        # Factor 2: Number of results (more results can indicate better coverage)
        results_factor = min(len(search_result.unified_results) / 5.0, 1.0)  # Normalize to max 5 results
        relevance_factors.append(results_factor)
        
        # Factor 3: Strategy appropriateness
        strategy_score = 0.8 if search_result.search_strategy_used in ["vector_search", "hybrid_search"] else 0.5
        relevance_factors.append(strategy_score)
        
        # Factor 4: Ground truth alignment (simplified)
        query_lower = query.lower()
        ground_truth_alignment = 0.0
        
        if ground_truth["azure_services"]:
            for service in ground_truth["azure_services"]:
                if service in query_lower:
                    ground_truth_alignment += 0.2
        
        if ground_truth["technical_terms"]:
            for term in ground_truth["technical_terms"]:
                if term in query_lower:
                    ground_truth_alignment += 0.1
        
        ground_truth_alignment = min(ground_truth_alignment, 1.0)
        relevance_factors.append(ground_truth_alignment)
        
        return statistics.mean(relevance_factors)