#!/usr/bin/env python3
"""
Agent 1 (Domain Intelligence) Comprehensive Validation
======================================================

Complete validation testing for Agent 1 (Domain Intelligence Agent) with schema compliance,
output quality analysis, and comparison against design requirements.

This script provides:
- Direct Agent 1 testing with real Azure services
- Complete output schema validation (26 required fields)
- Content quality assessment
- Performance metrics
- Comparison against filtered outputs
- Production readiness validation

Usage:
    python scripts/dataflow/00_agent1_validation.py
    python scripts/dataflow/00_agent1_validation.py --file custom_file.md
    python scripts/dataflow/00_agent1_validation.py --test-multiple
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

from agents.domain_intelligence.agent import run_domain_analysis
from agents.core.universal_models import UniversalDomainAnalysis


class Agent1Validator:
    """Comprehensive Agent 1 validation with schema compliance and quality assessment"""
    
    def __init__(self):
        self.results = {
            "test_runs": [],
            "summary": {},
            "compliance_analysis": {},
            "quality_metrics": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Required schema fields per UniversalDomainAnalysis
        self.required_top_level = [
            'domain_signature', 'content_type_confidence', 
            'analysis_timestamp', 'processing_time', 'data_source_path', 'analysis_reliability',
            'key_insights', 'adaptation_recommendations'
        ]
        
        self.required_characteristics = [
            'avg_document_length', 'document_count', 'vocabulary_richness', 'sentence_complexity',
            'most_frequent_terms', 'content_patterns', 'language_indicators', 
            'lexical_diversity', 'vocabulary_complexity_ratio', 'structural_consistency'
        ]
        
        self.required_processing_config = [
            'optimal_chunk_size', 'chunk_overlap_ratio', 'entity_confidence_threshold', 'relationship_density',
            'vector_search_weight', 'graph_search_weight', 'expected_extraction_quality', 'processing_complexity'
        ]

    async def validate_agent1_output(self, content: str, source_file: str = "test_content") -> Dict[str, Any]:
        """
        Run Agent 1 with content and perform comprehensive validation
        
        Args:
            content: Text content to analyze
            source_file: Source file name for tracking
            
        Returns:
            Complete validation results dictionary
        """
        print(f"\n🧪 AGENT 1 VALIDATION: {source_file}")
        print(f"{'=' * 60}")
        print(f"📄 Content length: {len(content)} characters")
        print(f"📝 Preview: {content[:150]}...")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Run Agent 1 directly
            print(f"\n🚀 Running Agent 1 with real Azure services...")
            result = await run_domain_analysis(content, detailed=True)
            processing_time = time.time() - start_time
            
            print(f"✅ Agent 1 completed in {processing_time:.2f}s")
            
            # Convert to dict for analysis
            output_dict = result.model_dump()
            
            # Perform comprehensive validation
            validation_result = self._validate_schema_compliance(output_dict)
            quality_result = self._assess_output_quality(output_dict, content)
            
            # Combine results
            test_result = {
                "source_file": source_file,
                "content_length": len(content),
                "processing_time": processing_time,
                "agent1_execution_time": output_dict.get("processing_time", 0),
                "schema_compliance": validation_result,
                "quality_assessment": quality_result,
                "complete_output": output_dict,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Display results
            self._display_validation_results(test_result)
            
            return test_result
            
        except Exception as e:
            error_result = {
                "source_file": source_file,
                "content_length": len(content),
                "processing_time": time.time() - start_time,
                "error": str(e),
                "success": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"❌ Agent 1 validation failed: {e}")
            return error_result

    def _validate_schema_compliance(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output against required schema fields"""
        
        # Check top-level fields
        present_top = [f for f in self.required_top_level if f in output]
        missing_top = [f for f in self.required_top_level if f not in output]
        
        # Check characteristics fields
        chars = output.get('characteristics', {})
        present_chars = [f for f in self.required_characteristics if f in chars]
        missing_chars = [f for f in self.required_characteristics if f not in chars]
        
        # Check processing_config fields
        config = output.get('processing_config', {})
        present_config = [f for f in self.required_processing_config if f in config]
        missing_config = [f for f in self.required_processing_config if f not in config]
        
        # Calculate compliance metrics
        total_present = len(present_top) + len(present_chars) + len(present_config)
        total_required = len(self.required_top_level) + len(self.required_characteristics) + len(self.required_processing_config)
        compliance_percentage = (total_present / total_required) * 100
        
        # Check for field name issues
        field_issues = []
        if 'vocabulary_complexity' in chars and 'vocabulary_complexity_ratio' not in chars:
            field_issues.append("Uses 'vocabulary_complexity' instead of 'vocabulary_complexity_ratio'")
        
        return {
            "total_compliance_percentage": compliance_percentage,
            "total_present": total_present,
            "total_required": total_required,
            "top_level_compliance": {
                "present": present_top,
                "missing": missing_top,
                "percentage": (len(present_top) / len(self.required_top_level)) * 100
            },
            "characteristics_compliance": {
                "present": present_chars,
                "missing": missing_chars,
                "percentage": (len(present_chars) / len(self.required_characteristics)) * 100
            },
            "processing_config_compliance": {
                "present": present_config,
                "missing": missing_config,
                "percentage": (len(present_config) / len(self.required_processing_config)) * 100
            },
            "field_name_issues": field_issues,
            "is_fully_compliant": compliance_percentage == 100.0 and len(field_issues) == 0
        }

    def _assess_output_quality(self, output: Dict[str, Any], original_content: str) -> Dict[str, Any]:
        """Assess the quality and meaningfulness of Agent 1 output"""
        
        quality_metrics = {
            "content_analysis_quality": "unknown",
            "processing_config_quality": "unknown", 
            "insights_quality": "unknown",
            "overall_quality": "unknown",
            "quality_score": 0
        }
        
        score = 0
        max_score = 10
        
        # Check content analysis quality
        chars = output.get('characteristics', {})
        
        # 1. Meaningful most_frequent_terms (2 points)
        terms = chars.get('most_frequent_terms', [])
        if terms and len(terms) >= 3 and all(len(term) > 2 for term in terms):
            score += 2
            quality_metrics["has_meaningful_terms"] = True
        
        # 2. Realistic sentence_complexity (1 point)  
        sentence_complexity = chars.get('sentence_complexity', 0)
        if 5 <= sentence_complexity <= 50:  # Realistic range
            score += 1
            quality_metrics["realistic_sentence_complexity"] = True
            
        # 3. Content patterns identified (1 point)
        patterns = chars.get('content_patterns', [])
        if patterns and len(patterns) >= 1:
            score += 1
            quality_metrics["identifies_content_patterns"] = True
            
        # 4. Dynamic processing config (2 points)
        config = output.get('processing_config', {})
        chunk_size = config.get('optimal_chunk_size', 0)
        if 100 <= chunk_size <= 4000:  # Within valid range
            score += 2
            quality_metrics["dynamic_chunk_sizing"] = True
            
        # 5. Meaningful insights (2 points)
        insights = output.get('key_insights', [])
        if insights and len(insights) >= 2 and all(len(insight) > 20 for insight in insights):
            score += 2
            quality_metrics["meaningful_insights"] = True
            
        # 6. Practical recommendations (2 points)
        recommendations = output.get('adaptation_recommendations', [])
        if recommendations and len(recommendations) >= 2:
            score += 2
            quality_metrics["practical_recommendations"] = True
        
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
            "overall_quality": overall_quality
        })
        
        return quality_metrics

    def _display_validation_results(self, result: Dict[str, Any]):
        """Display formatted validation results"""
        
        print(f"\n📊 VALIDATION RESULTS")
        print(f"{'=' * 40}")
        
        # Schema compliance
        compliance = result["schema_compliance"]
        print(f"🔍 Schema Compliance: {compliance['total_compliance_percentage']:.1f}% ({compliance['total_present']}/{compliance['total_required']})")
        
        if compliance["is_fully_compliant"]:
            print(f"✅ FULLY COMPLIANT: All required fields present with correct names")
        else:
            print(f"❌ Compliance Issues:")
            for category in ["top_level_compliance", "characteristics_compliance", "processing_config_compliance"]:
                cat_data = compliance[category]
                if cat_data["missing"]:
                    print(f"  • {category}: Missing {cat_data['missing']}")
            if compliance["field_name_issues"]:
                print(f"  • Field name issues: {compliance['field_name_issues']}")
        
        # Quality assessment
        quality = result["quality_assessment"]
        print(f"\n🎯 Output Quality: {quality['overall_quality'].upper()} ({quality['quality_percentage']:.1f}%)")
        print(f"   Quality score: {quality['quality_score']}/{quality['max_score']}")
        
        # Key metrics
        output = result["complete_output"]
        print(f"\n📋 Key Output Metrics:")
        print(f"   Domain signature: {output.get('domain_signature', 'N/A')}")
        print(f"   Processing time: {result.get('agent1_execution_time', 0):.1f}s")
        print(f"   Content confidence: {output.get('content_type_confidence', 0):.2f}")
        print(f"   Analysis reliability: {output.get('analysis_reliability', 0):.2f}")
        
        # Sample outputs
        chars = output.get('characteristics', {})
        if chars.get('most_frequent_terms'):
            print(f"   Top terms: {chars['most_frequent_terms'][:5]}")
        if chars.get('content_patterns'):
            print(f"   Content patterns: {chars['content_patterns']}")

    async def run_multiple_file_validation(self, data_dir: str, max_files: int = 5):
        """Run validation on multiple files from data directory"""
        
        print(f"🧪 MULTIPLE FILE VALIDATION")
        print(f"{'=' * 50}")
        print(f"📁 Data directory: {data_dir}")
        print(f"📊 Max files to test: {max_files}")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return
        
        # Find files to test
        test_files = []
        for pattern in ["*.md", "*.txt"]:
            test_files.extend(list(data_path.glob(pattern)))
        
        if not test_files:
            print(f"❌ No files found in {data_dir}")
            return
            
        test_files = test_files[:max_files]
        print(f"📋 Testing {len(test_files)} files")
        
        # Run validation on each file
        all_results = []
        for i, file_path in enumerate(test_files, 1):
            print(f"\n{'=' * 20} FILE {i}/{len(test_files)} {'=' * 20}")
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if len(content) < 100:  # Skip very short files
                    print(f"⏭️  Skipping {file_path.name}: too short ({len(content)} chars)")
                    continue
                    
                result = await self.validate_agent1_output(content, file_path.name)
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
        
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
        avg_quality = sum(r["quality_assessment"]["quality_percentage"] for r in successful_results) / len(successful_results)
        avg_processing_time = sum(r["processing_time"] for r in successful_results) / len(successful_results)
        
        fully_compliant_count = sum(1 for r in successful_results if r["schema_compliance"]["is_fully_compliant"])
        high_quality_count = sum(1 for r in successful_results if r["quality_assessment"]["overall_quality"] in ["excellent", "good"])
        
        print(f"📈 Overall Performance:")
        print(f"   Tests run: {len(successful_results)}")
        print(f"   Average compliance: {avg_compliance:.1f}%")
        print(f"   Average quality: {avg_quality:.1f}%")
        print(f"   Average processing time: {avg_processing_time:.2f}s")
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
            
        print(f"\n🎯 OVERALL AGENT 1 ASSESSMENT: {overall_rating}")
        
        # Save detailed results
        report_file = Path("agent1_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "tests_run": len(successful_results),
                    "average_compliance": avg_compliance,
                    "average_quality": avg_quality,
                    "average_processing_time": avg_processing_time,
                    "fully_compliant_count": fully_compliant_count,
                    "high_quality_count": high_quality_count,
                    "overall_rating": overall_rating
                },
                "detailed_results": results
            }, f, indent=2, default=str)
        
        print(f"💾 Detailed report saved to: {report_file}")


async def main():
    """Main function with command line argument handling"""
    parser = argparse.ArgumentParser(description="Agent 1 (Domain Intelligence) Comprehensive Validation")
    parser.add_argument("--file", type=str, help="Specific file to test")
    parser.add_argument("--test-multiple", action="store_true", help="Test multiple files")
    parser.add_argument("--data-dir", type=str, default="/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output", 
                       help="Data directory for multiple file testing")
    parser.add_argument("--max-files", type=int, default=5, help="Maximum number of files to test")
    
    args = parser.parse_args()
    
    print("🧪 AGENT 1 (DOMAIN INTELLIGENCE) COMPREHENSIVE VALIDATION")
    print("=" * 65)
    print("Purpose: Validate Agent 1 schema compliance and output quality")
    print("Scope: Real Azure services with complete output analysis")
    
    validator = Agent1Validator()
    
    if args.test_multiple:
        # Test multiple files
        await validator.run_multiple_file_validation(args.data_dir, args.max_files)
        
    elif args.file:
        # Test specific file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            return
            
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        result = await validator.validate_agent1_output(content, file_path.name)
        
    else:
        # Test with default sample
        data_dir = Path("/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output")
        if data_dir.exists():
            sample_files = list(data_dir.glob("*.md"))
            if sample_files:
                sample_file = sample_files[0]
                content = sample_file.read_text(encoding='utf-8', errors='ignore')
                result = await validator.validate_agent1_output(content, sample_file.name)
            else:
                print("❌ No sample files found for testing")
        else:
            print(f"❌ Sample data directory not found: {data_dir}")


if __name__ == "__main__":
    asyncio.run(main())