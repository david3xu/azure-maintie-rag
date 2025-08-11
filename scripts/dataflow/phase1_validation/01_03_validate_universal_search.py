#!/usr/bin/env python3
"""
Agent 3 (Universal Search) Comprehensive Validation
===================================================

Complete validation testing for Agent 3 (Universal Search Agent) with schema compliance,
search quality analysis, and multi-modal search strategy validation.

This script provides:
- Direct Agent 3 testing with real Azure services
- Complete MultiModalSearchResult schema validation
- Search quality and relevance assessment
- Multi-modal coordination validation (Vector + Graph + GNN)
- Search strategy effectiveness analysis

Usage:
    python scripts/dataflow/00_agent3_validation.py
    python scripts/dataflow/00_agent3_validation.py --query "custom search query"
    python scripts/dataflow/00_agent3_validation.py --test-multiple
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_search.agent import run_universal_search
from agents.core.universal_models import MultiModalSearchResult


class Agent3Validator:
    """Comprehensive Agent 3 validation with schema compliance and search quality assessment"""
    
    def __init__(self):
        self.results = {
            "test_runs": [],
            "summary": {},
            "search_analysis": {},
            "quality_metrics": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Required schema fields per MultiModalSearchResult
        self.required_top_level = [
            'unified_results', 'total_results_found', 'search_strategy_used', 
            'search_confidence', 'processing_time_seconds',
            'vector_results', 'graph_results', 'gnn_results'
        ]
        
        # Unified result validation fields
        self.required_result_fields = [
            'title', 'content', 'score', 'source', 'metadata'
        ]
        
        # Search strategy validation
        self.valid_strategies = [
            'vector_only', 'graph_only', 'gnn_only',
            'vector_graph', 'vector_gnn', 'graph_gnn', 
            'tri_modal', 'adaptive_hybrid'
        ]

    async def validate_agent3_output(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Run Agent 3 with query and perform comprehensive validation
        
        Args:
            query: Search query to analyze
            max_results: Maximum number of results to retrieve
            
        Returns:
            Complete validation results dictionary
        """
        print(f"\n🧪 AGENT 3 VALIDATION")
        print(f"{'=' * 60}")
        print(f"🔍 Search query: {query}")
        print(f"📊 Max results: {max_results}")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Run Agent 3 directly
            print(f"\n🚀 Running Agent 3 with real Azure services...")
            result = await run_universal_search(
                query,
                max_results=max_results,
                use_domain_analysis=True
            )
            processing_time = time.time() - start_time
            
            print(f"✅ Agent 3 completed in {processing_time:.2f}s")
            
            # Convert to dict for analysis
            output_dict = result.model_dump()
            
            # Perform comprehensive validation
            validation_result = self._validate_schema_compliance(output_dict)
            quality_result = self._assess_search_quality(output_dict, query)
            strategy_result = self._assess_search_strategy(output_dict)
            
            # Combine results
            test_result = {
                "search_query": query,
                "max_results_requested": max_results,
                "processing_time": processing_time,
                "agent3_execution_time": output_dict.get("processing_time_seconds", 0),
                "schema_compliance": validation_result,
                "search_quality": quality_result,
                "strategy_assessment": strategy_result,
                "complete_output": output_dict,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Display results
            self._display_validation_results(test_result)
            
            return test_result
            
        except Exception as e:
            error_result = {
                "search_query": query,
                "max_results_requested": max_results,
                "processing_time": time.time() - start_time,
                "error": str(e),
                "success": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"❌ Agent 3 validation failed: {e}")
            return error_result

    def _validate_schema_compliance(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output against required MultiModalSearchResult schema fields"""
        
        # Check top-level fields
        present_top = [f for f in self.required_top_level if f in output]
        missing_top = [f for f in self.required_top_level if f not in output]
        
        # Check unified results structure
        unified_results = output.get('unified_results', [])
        results_compliance = self._validate_results_structure(unified_results)
        
        # Check multi-modal results existence
        vector_results = output.get('vector_results', [])
        graph_results = output.get('graph_results', [])
        gnn_results = output.get('gnn_results', [])
        
        modality_compliance = {
            "has_vector_results": len(vector_results) > 0,
            "has_graph_results": len(graph_results) > 0,
            "has_gnn_results": len(gnn_results) > 0,
            "multi_modal_count": sum([len(vector_results) > 0, len(graph_results) > 0, len(gnn_results) > 0])
        }
        
        # Calculate compliance metrics
        total_present = len(present_top)
        total_required = len(self.required_top_level)
        
        # Add results structure compliance
        if results_compliance["valid_structure"]:
            total_present += 1
        total_required += 1
        
        compliance_percentage = (total_present / total_required) * 100
        
        return {
            "total_compliance_percentage": compliance_percentage,
            "total_present": total_present,
            "total_required": total_required,
            "top_level_compliance": {
                "present": present_top,
                "missing": missing_top,
                "percentage": (len(present_top) / len(self.required_top_level)) * 100
            },
            "results_compliance": results_compliance,
            "modality_compliance": modality_compliance,
            "is_fully_compliant": compliance_percentage == 100.0
        }

    def _validate_results_structure(self, unified_results: List[Dict]) -> Dict[str, Any]:
        """Validate unified search results structure and field completeness"""
        
        if not unified_results:
            return {
                "valid_structure": True,  # Empty results can be valid
                "result_count": 0,
                "field_compliance": 100,
                "issues": []
            }
        
        valid_results = 0
        total_fields = 0
        present_fields = 0
        issues = []
        
        for i, result in enumerate(unified_results):
            result_valid = True
            for field in self.required_result_fields:
                total_fields += 1
                if field in result:
                    present_fields += 1
                else:
                    result_valid = False
                    issues.append(f"Result {i}: missing '{field}' field")
            
            if result_valid:
                valid_results += 1
                
        field_compliance = (present_fields / total_fields * 100) if total_fields > 0 else 100
        
        return {
            "valid_structure": len(issues) == 0,
            "result_count": len(unified_results),
            "valid_results": valid_results,
            "field_compliance": field_compliance,
            "issues": issues
        }

    def _assess_search_quality(self, output: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Assess the quality and relevance of search results"""
        
        quality_metrics = {
            "result_relevance": "unknown",
            "coverage_quality": "unknown",
            "confidence_quality": "unknown",
            "overall_search_quality": "unknown",
            "quality_score": 0
        }
        
        score = 0
        max_score = 10
        
        unified_results = output.get('unified_results', [])
        total_results = output.get('total_results_found', 0)
        search_confidence = output.get('search_confidence', 0)
        
        # 1. Result quantity quality (2 points)
        if total_results >= 3:
            score += 2
            quality_metrics["sufficient_results"] = True
        elif total_results >= 1:
            score += 1
            quality_metrics["minimal_results"] = True
        
        # 2. Search confidence quality (2 points)
        if search_confidence >= 0.8:
            score += 2
            quality_metrics["high_confidence"] = True
        elif search_confidence >= 0.6:
            score += 1
            quality_metrics["moderate_confidence"] = True
            
        # 3. Result score distribution (2 points)
        if unified_results:
            scores = [r.get('score', 0) for r in unified_results]
            avg_score = sum(scores) / len(scores)
            if avg_score >= 0.7:
                score += 2
                quality_metrics["high_result_scores"] = True
            elif avg_score >= 0.5:
                score += 1
                quality_metrics["moderate_result_scores"] = True
        
        # 4. Content quality (2 points)
        if unified_results:
            # Check if results have meaningful content
            content_lengths = [len(r.get('content', '')) for r in unified_results]
            avg_content_length = sum(content_lengths) / len(content_lengths)
            if avg_content_length >= 100:
                score += 2
                quality_metrics["meaningful_content"] = True
            elif avg_content_length >= 50:
                score += 1
                quality_metrics["minimal_content"] = True
        
        # 5. Multi-modal coordination (1 point)
        vector_results = output.get('vector_results', [])
        graph_results = output.get('graph_results', [])
        gnn_results = output.get('gnn_results', [])
        active_modalities = sum([len(vector_results) > 0, len(graph_results) > 0, len(gnn_results) > 0])
        
        if active_modalities >= 2:
            score += 1
            quality_metrics["multi_modal_coordination"] = True
        
        # 6. Processing performance (1 point)
        processing_time = output.get('processing_time_seconds', 0)
        if 0.5 <= processing_time <= 10:  # Reasonable processing time
            score += 1
            quality_metrics["reasonable_processing_time"] = True
        
        # Calculate quality ratings
        quality_percentage = (score / max_score) * 100
        
        if quality_percentage >= 90:
            overall_quality = "excellent"
        elif quality_percentage >= 75:
            overall_quality = "good"
        elif quality_percentage >= 60:
            overall_quality = "moderate"
        else:
            overall_quality = "poor"
            
        quality_metrics.update({
            "quality_score": score,
            "max_score": max_score,
            "quality_percentage": quality_percentage,
            "overall_search_quality": overall_quality,
            "total_results_found": total_results,
            "unified_results_count": len(unified_results),
            "search_confidence": search_confidence,
            "active_modalities": active_modalities,
            "average_result_score": sum(r.get('score', 0) for r in unified_results) / len(unified_results) if unified_results else 0
        })
        
        return quality_metrics

    def _assess_search_strategy(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Assess search strategy effectiveness and appropriateness"""
        
        strategy_used = output.get('search_strategy_used', 'unknown')
        vector_results = output.get('vector_results', [])
        graph_results = output.get('graph_results', [])
        gnn_results = output.get('gnn_results', [])
        
        strategy_metrics = {
            "strategy_used": strategy_used,
            "strategy_validity": strategy_used in self.valid_strategies,
            "modality_coordination": {
                "vector_active": len(vector_results) > 0,
                "graph_active": len(graph_results) > 0,
                "gnn_active": len(gnn_results) > 0
            },
            "strategy_effectiveness": "unknown"
        }
        
        # Assess strategy effectiveness based on results
        total_results = len(vector_results) + len(graph_results) + len(gnn_results)
        unified_count = len(output.get('unified_results', []))
        
        if total_results > 0 and unified_count > 0:
            coordination_ratio = unified_count / total_results
            if coordination_ratio >= 0.7:
                strategy_metrics["strategy_effectiveness"] = "high"
            elif coordination_ratio >= 0.4:
                strategy_metrics["strategy_effectiveness"] = "moderate"
            else:
                strategy_metrics["strategy_effectiveness"] = "low"
        
        # Check strategy alignment with modality usage
        expected_modalities = []
        if 'vector' in strategy_used:
            expected_modalities.append('vector')
        if 'graph' in strategy_used:
            expected_modalities.append('graph')
        if 'gnn' in strategy_used:
            expected_modalities.append('gnn')
        if 'tri_modal' in strategy_used or 'hybrid' in strategy_used:
            expected_modalities = ['vector', 'graph', 'gnn']
        
        actual_modalities = []
        if len(vector_results) > 0:
            actual_modalities.append('vector')
        if len(graph_results) > 0:
            actual_modalities.append('graph')
        if len(gnn_results) > 0:
            actual_modalities.append('gnn')
        
        strategy_metrics["strategy_alignment"] = set(expected_modalities).issubset(set(actual_modalities))
        
        return strategy_metrics

    def _display_validation_results(self, result: Dict[str, Any]):
        """Display formatted validation results"""
        
        print(f"\n📊 VALIDATION RESULTS")
        print(f"{'=' * 40}")
        
        # Schema compliance
        compliance = result["schema_compliance"]
        print(f"🔍 Schema Compliance: {compliance['total_compliance_percentage']:.1f}% ({compliance['total_present']}/{compliance['total_required']})")
        
        if compliance["is_fully_compliant"]:
            print(f"✅ FULLY COMPLIANT: All required fields and structures valid")
        else:
            print(f"❌ Compliance Issues:")
            if compliance["top_level_compliance"]["missing"]:
                print(f"  • Missing top-level: {compliance['top_level_compliance']['missing']}")
            if compliance["results_compliance"]["issues"]:
                print(f"  • Result structure issues: {len(compliance['results_compliance']['issues'])} found")
        
        # Search quality
        quality = result["search_quality"]
        print(f"\n🎯 Search Quality: {quality['overall_search_quality'].upper()} ({quality['quality_percentage']:.1f}%)")
        print(f"   Quality score: {quality['quality_score']}/{quality['max_score']}")
        
        # Search metrics
        output = result["complete_output"]
        print(f"\n📋 Search Metrics:")
        print(f"   Query: {result['search_query']}")
        print(f"   Total results found: {quality['total_results_found']}")
        print(f"   Unified results: {quality['unified_results_count']}")
        print(f"   Search confidence: {quality['search_confidence']:.2f}")
        print(f"   Active modalities: {quality['active_modalities']}/3")
        print(f"   Processing time: {result.get('agent3_execution_time', 0):.1f}s")
        
        # Strategy assessment
        strategy = result["strategy_assessment"]
        print(f"\n🎭 Search Strategy:")
        print(f"   Strategy used: {strategy['strategy_used']}")
        print(f"   Strategy valid: {strategy['strategy_validity']}")
        print(f"   Strategy effectiveness: {strategy['strategy_effectiveness']}")
        print(f"   Strategy alignment: {strategy['strategy_alignment']}")
        
        # Modality breakdown
        modality = compliance["modality_compliance"]
        print(f"   Vector results: {'✅' if modality['has_vector_results'] else '❌'}")
        print(f"   Graph results: {'✅' if modality['has_graph_results'] else '❌'}")
        print(f"   GNN results: {'✅' if modality['has_gnn_results'] else '❌'}")
        
        # Sample results
        unified_results = output.get('unified_results', [])
        if unified_results:
            print(f"\n🏆 Top Search Results:")
            for i, res in enumerate(unified_results[:2], 1):
                print(f"   {i}. {res.get('title', 'N/A')[:50]}... (score: {res.get('score', 0):.2f}, source: {res.get('source', 'N/A')})")

    async def run_multiple_query_validation(self, queries: List[str]):
        """Run validation on multiple search queries"""
        
        print(f"🧪 MULTIPLE QUERY VALIDATION")
        print(f"{'=' * 50}")
        print(f"📋 Testing {len(queries)} queries")
        
        # Run validation on each query
        all_results = []
        for i, query in enumerate(queries, 1):
            print(f"\n{'=' * 20} QUERY {i}/{len(queries)} {'=' * 20}")
            
            try:
                result = await self.validate_agent3_output(query, max_results=3)
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing query '{query}': {e}")
        
        # Generate summary
        if all_results:
            self._generate_summary_report(all_results)
            
        return all_results

    def _generate_summary_report(self, results: List[Dict[str, Any]]):
        """Generate summary report from multiple validation results"""
        
        print(f"\n📊 SUMMARY REPORT")
        print(f"{'=' * 40}")
        
        successful_results = [r for r in results if "schema_compliance" in r]
        
        if not successful_results:
            print(f"❌ No successful validations to summarize")
            return
        
        # Calculate averages
        avg_compliance = sum(r["schema_compliance"]["total_compliance_percentage"] for r in successful_results) / len(successful_results)
        avg_quality = sum(r["search_quality"]["quality_percentage"] for r in successful_results) / len(successful_results)
        avg_processing_time = sum(r["processing_time"] for r in successful_results) / len(successful_results)
        avg_results_found = sum(r["search_quality"]["total_results_found"] for r in successful_results) / len(successful_results)
        avg_search_confidence = sum(r["search_quality"]["search_confidence"] for r in successful_results) / len(successful_results)
        
        fully_compliant_count = sum(1 for r in successful_results if r["schema_compliance"]["is_fully_compliant"])
        high_quality_count = sum(1 for r in successful_results if r["search_quality"]["overall_search_quality"] in ["excellent", "good"])
        
        print(f"📈 Overall Performance:")
        print(f"   Tests run: {len(successful_results)}")
        print(f"   Average compliance: {avg_compliance:.1f}%")
        print(f"   Average search quality: {avg_quality:.1f}%")
        print(f"   Average processing time: {avg_processing_time:.2f}s")
        print(f"   Average results per query: {avg_results_found:.1f}")
        print(f"   Average search confidence: {avg_search_confidence:.2f}")
        print(f"   Fully compliant tests: {fully_compliant_count}/{len(successful_results)} ({fully_compliant_count/len(successful_results)*100:.1f}%)")
        print(f"   High quality tests: {high_quality_count}/{len(successful_results)} ({high_quality_count/len(successful_results)*100:.1f}%)")
        
        # Overall assessment
        if avg_compliance >= 95 and avg_quality >= 80:
            overall_rating = "🎉 EXCELLENT"
        elif avg_compliance >= 85 and avg_quality >= 70:
            overall_rating = "✅ GOOD"
        elif avg_compliance >= 70:
            overall_rating = "⚠️  MODERATE"
        else:
            overall_rating = "❌ NEEDS IMPROVEMENT"
            
        print(f"\n🎯 OVERALL AGENT 3 ASSESSMENT: {overall_rating}")
        
        # Save detailed results
        report_file = Path("agent3_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "tests_run": len(successful_results),
                    "average_compliance": avg_compliance,
                    "average_quality": avg_quality,
                    "average_processing_time": avg_processing_time,
                    "average_results_found": avg_results_found,
                    "average_search_confidence": avg_search_confidence,
                    "fully_compliant_count": fully_compliant_count,
                    "high_quality_count": high_quality_count,
                    "overall_rating": overall_rating
                },
                "detailed_results": results
            }, f, indent=2, default=str)
        
        print(f"💾 Detailed report saved to: {report_file}")


async def main():
    """Main function with command line argument handling"""
    parser = argparse.ArgumentParser(description="Agent 3 (Universal Search) Comprehensive Validation")
    parser.add_argument("--query", type=str, help="Specific query to test")
    parser.add_argument("--test-multiple", action="store_true", help="Test multiple queries")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum results per query")
    
    args = parser.parse_args()
    
    print("🧪 AGENT 3 (UNIVERSAL SEARCH) COMPREHENSIVE VALIDATION")
    print("=" * 65)
    print("Purpose: Validate Agent 3 schema compliance and search quality")
    print("Scope: Real Azure services with multi-modal search analysis")
    
    validator = Agent3Validator()
    
    if args.test_multiple:
        # Test multiple queries
        demo_queries = [
            "Azure Cosmos DB performance optimization",
            "machine learning model deployment strategies",
            "knowledge graph construction techniques"
        ]
        await validator.run_multiple_query_validation(demo_queries)
        
    elif args.query:
        # Test specific query
        result = await validator.validate_agent3_output(args.query, args.max_results)
        
    else:
        # Test with default query
        default_query = "Azure AI services integration patterns"
        result = await validator.validate_agent3_output(default_query, args.max_results)


if __name__ == "__main__":
    asyncio.run(main())