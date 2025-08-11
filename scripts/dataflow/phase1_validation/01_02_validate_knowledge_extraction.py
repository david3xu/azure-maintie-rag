#!/usr/bin/env python3
"""
Agent 2 (Knowledge Extraction) Comprehensive Validation
=======================================================

Complete validation testing for Agent 2 (Knowledge Extraction Agent) with schema compliance,
output quality analysis, and entity/relationship extraction validation.

This script provides:
- Direct Agent 2 testing with real Azure services
- Complete ExtractionResult schema validation
- Entity and relationship quality assessment
- Processing signature validation
- Performance metrics and production readiness testing

Usage:
    python scripts/dataflow/00_agent2_validation.py
    python scripts/dataflow/00_agent2_validation.py --file custom_file.md
    python scripts/dataflow/00_agent2_validation.py --test-multiple
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

from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.core.universal_models import ExtractionResult


class Agent2Validator:
    """Comprehensive Agent 2 validation with schema compliance and extraction quality assessment"""
    
    def __init__(self):
        self.results = {
            "test_runs": [],
            "summary": {},
            "extraction_analysis": {},
            "quality_metrics": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Required schema fields per ExtractionResult
        self.required_top_level = [
            'entities', 'relationships', 'processing_signature', 
            'extraction_confidence', 'processing_time', 'extracted_concepts'
        ]
        
        # Entity validation fields
        self.required_entity_fields = [
            'text', 'type', 'confidence', 'context', 'positions'
        ]
        
        # Relationship validation fields  
        self.required_relationship_fields = [
            'source', 'target', 'relation', 'confidence', 'metadata'
        ]

    async def validate_agent2_output(self, content: str, source_file: str = "test_content") -> Dict[str, Any]:
        """
        Run Agent 2 with content and perform comprehensive validation
        
        Args:
            content: Text content to analyze
            source_file: Source file name for tracking
            
        Returns:
            Complete validation results dictionary
        """
        print(f"\n🧪 AGENT 2 VALIDATION: {source_file}")
        print(f"{'=' * 60}")
        print(f"📄 Content length: {len(content)} characters")
        print(f"📝 Preview: {content[:150]}...")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Run Agent 2 directly
            print(f"\n🚀 Running Agent 2 with real Azure services...")
            result = await run_knowledge_extraction(
                content,
                use_domain_analysis=True
            )
            processing_time = time.time() - start_time
            
            print(f"✅ Agent 2 completed in {processing_time:.2f}s")
            
            # Convert to dict for analysis
            output_dict = result.model_dump()
            
            # Perform comprehensive validation
            validation_result = self._validate_schema_compliance(output_dict)
            quality_result = self._assess_extraction_quality(output_dict, content)
            
            # Combine results
            test_result = {
                "source_file": source_file,
                "content_length": len(content),
                "processing_time": processing_time,
                "agent2_execution_time": output_dict.get("processing_time", 0),
                "schema_compliance": validation_result,
                "extraction_quality": quality_result,
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
            print(f"❌ Agent 2 validation failed: {e}")
            return error_result

    def _validate_schema_compliance(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output against required ExtractionResult schema fields"""
        
        # Check top-level fields
        present_top = [f for f in self.required_top_level if f in output]
        missing_top = [f for f in self.required_top_level if f not in output]
        
        # Check entity structure
        entities = output.get('entities', [])
        entity_compliance = self._validate_entity_structure(entities)
        
        # Check relationship structure
        relationships = output.get('relationships', [])
        relationship_compliance = self._validate_relationship_structure(relationships)
        
        # Calculate compliance metrics
        total_present = len(present_top)
        total_required = len(self.required_top_level)
        
        # Add entity/relationship structure compliance
        if entity_compliance["valid_structure"]:
            total_present += 1
        if relationship_compliance["valid_structure"]:
            total_present += 1
        total_required += 2  # For entity and relationship structures
        
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
            "entity_compliance": entity_compliance,
            "relationship_compliance": relationship_compliance,
            "is_fully_compliant": compliance_percentage == 100.0
        }

    def _validate_entity_structure(self, entities: List[Dict]) -> Dict[str, Any]:
        """Validate entity structure and field completeness"""
        
        if not entities:
            return {
                "valid_structure": False,
                "entity_count": 0,
                "field_compliance": 0,
                "issues": ["No entities extracted"]
            }
        
        valid_entities = 0
        total_fields = 0
        present_fields = 0
        issues = []
        
        for i, entity in enumerate(entities):
            entity_valid = True
            for field in self.required_entity_fields:
                total_fields += 1
                if field in entity:
                    present_fields += 1
                else:
                    entity_valid = False
                    issues.append(f"Entity {i}: missing '{field}' field")
            
            if entity_valid:
                valid_entities += 1
                
        field_compliance = (present_fields / total_fields * 100) if total_fields > 0 else 0
        
        return {
            "valid_structure": len(issues) == 0,
            "entity_count": len(entities),
            "valid_entities": valid_entities,
            "field_compliance": field_compliance,
            "issues": issues
        }

    def _validate_relationship_structure(self, relationships: List[Dict]) -> Dict[str, Any]:
        """Validate relationship structure and field completeness"""
        
        if not relationships:
            return {
                "valid_structure": True,  # Empty relationships list is acceptable
                "relationship_count": 0,
                "field_compliance": 100,
                "issues": []
            }
        
        valid_relationships = 0
        total_fields = 0
        present_fields = 0
        issues = []
        
        for i, rel in enumerate(relationships):
            rel_valid = True
            for field in self.required_relationship_fields:
                total_fields += 1
                if field in rel:
                    present_fields += 1
                else:
                    rel_valid = False
                    issues.append(f"Relationship {i}: missing '{field}' field")
            
            if rel_valid:
                valid_relationships += 1
                
        field_compliance = (present_fields / total_fields * 100) if total_fields > 0 else 100
        
        return {
            "valid_structure": len(issues) == 0,
            "relationship_count": len(relationships),
            "valid_relationships": valid_relationships,
            "field_compliance": field_compliance,
            "issues": issues
        }

    def _assess_extraction_quality(self, output: Dict[str, Any], original_content: str) -> Dict[str, Any]:
        """Assess the quality and meaningfulness of Agent 2 extraction results"""
        
        quality_metrics = {
            "entity_quality": "unknown",
            "relationship_quality": "unknown",
            "processing_signature_quality": "unknown",
            "overall_extraction_quality": "unknown",
            "quality_score": 0
        }
        
        score = 0
        max_score = 10
        
        entities = output.get('entities', [])
        relationships = output.get('relationships', [])
        
        # 1. Entity extraction quality (4 points total)
        if entities:
            # Meaningful entity count (2 points)
            if len(entities) >= 3:
                score += 2
                quality_metrics["sufficient_entities"] = True
                
            # Entity confidence quality (1 point)
            avg_entity_confidence = sum(e.get('confidence', 0) for e in entities) / len(entities)
            if avg_entity_confidence >= 0.6:
                score += 1
                quality_metrics["good_entity_confidence"] = True
                
            # Entity type diversity (1 point)
            entity_types = set(e.get('type', 'unknown') for e in entities)
            if len(entity_types) >= 2:
                score += 1
                quality_metrics["diverse_entity_types"] = True
        
        # 2. Relationship extraction quality (3 points total)
        if relationships:
            # Relationship presence (1 point)
            if len(relationships) >= 1:
                score += 1
                quality_metrics["has_relationships"] = True
                
            # Relationship confidence (1 point)
            avg_rel_confidence = sum(r.get('confidence', 0) for r in relationships) / len(relationships)
            if avg_rel_confidence >= 0.6:
                score += 1
                quality_metrics["good_relationship_confidence"] = True
                
            # Relationship diversity (1 point)
            rel_types = set(r.get('relation', 'unknown') for r in relationships)
            if len(rel_types) >= 2:
                score += 1
                quality_metrics["diverse_relationship_types"] = True
        
        # 3. Processing signature quality (1 point)
        processing_signature = output.get('processing_signature', '')
        if processing_signature and len(processing_signature) > 10:
            score += 1
            quality_metrics["meaningful_processing_signature"] = True
            
        # 4. Extraction confidence (1 point)
        extraction_confidence = output.get('extraction_confidence', 0)
        if extraction_confidence >= 0.7:
            score += 1
            quality_metrics["high_extraction_confidence"] = True
            
        # 5. Processing performance (1 point)
        processing_time = output.get('processing_time', 0)
        if 0.5 <= processing_time <= 30:  # Reasonable processing time
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
            "overall_extraction_quality": overall_quality,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "average_entity_confidence": avg_entity_confidence if entities else 0,
            "average_relationship_confidence": avg_rel_confidence if relationships else 0
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
            print(f"✅ FULLY COMPLIANT: All required fields and structures valid")
        else:
            print(f"❌ Compliance Issues:")
            if compliance["top_level_compliance"]["missing"]:
                print(f"  • Missing top-level: {compliance['top_level_compliance']['missing']}")
            if compliance["entity_compliance"]["issues"]:
                print(f"  • Entity issues: {len(compliance['entity_compliance']['issues'])} found")
            if compliance["relationship_compliance"]["issues"]:
                print(f"  • Relationship issues: {len(compliance['relationship_compliance']['issues'])} found")
        
        # Extraction quality
        quality = result["extraction_quality"]
        print(f"\n🎯 Extraction Quality: {quality['overall_extraction_quality'].upper()} ({quality['quality_percentage']:.1f}%)")
        print(f"   Quality score: {quality['quality_score']}/{quality['max_score']}")
        
        # Extraction metrics
        output = result["complete_output"]
        print(f"\n📋 Extraction Metrics:")
        print(f"   Entities extracted: {quality['entity_count']}")
        print(f"   Relationships extracted: {quality['relationship_count']}")
        print(f"   Processing signature: {output.get('processing_signature', 'N/A')}")
        print(f"   Extraction confidence: {output.get('extraction_confidence', 0):.2f}")
        print(f"   Processing time: {result.get('agent2_execution_time', 0):.1f}s")
        
        # Sample extractions
        entities = output.get('entities', [])
        if entities:
            print(f"   Top entities: {[e.get('text', 'N/A') for e in entities[:3]]}")
        
        relationships = output.get('relationships', [])
        if relationships:
            rel_samples = []
            for r in relationships[:2]:
                rel_samples.append(f"{r.get('source', 'N/A')}-[{r.get('relation', 'N/A')}]->{r.get('target', 'N/A')}")
            print(f"   Sample relationships: {rel_samples}")

    async def run_multiple_file_validation(self, data_dir: str, max_files: int = 3):
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
                if len(content) < 200:  # Skip very short files
                    print(f"⏭️  Skipping {file_path.name}: too short ({len(content)} chars)")
                    continue
                    
                result = await self.validate_agent2_output(content, file_path.name)
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
        avg_quality = sum(r["extraction_quality"]["quality_percentage"] for r in successful_results) / len(successful_results)
        avg_processing_time = sum(r["processing_time"] for r in successful_results) / len(successful_results)
        avg_entities = sum(r["extraction_quality"]["entity_count"] for r in successful_results) / len(successful_results)
        avg_relationships = sum(r["extraction_quality"]["relationship_count"] for r in successful_results) / len(successful_results)
        
        fully_compliant_count = sum(1 for r in successful_results if r["schema_compliance"]["is_fully_compliant"])
        high_quality_count = sum(1 for r in successful_results if r["extraction_quality"]["overall_extraction_quality"] in ["excellent", "good"])
        
        print(f"📈 Overall Performance:")
        print(f"   Tests run: {len(successful_results)}")
        print(f"   Average compliance: {avg_compliance:.1f}%")
        print(f"   Average extraction quality: {avg_quality:.1f}%")
        print(f"   Average processing time: {avg_processing_time:.2f}s")
        print(f"   Average entities per test: {avg_entities:.1f}")
        print(f"   Average relationships per test: {avg_relationships:.1f}")
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
            
        print(f"\n🎯 OVERALL AGENT 2 ASSESSMENT: {overall_rating}")
        
        # Save detailed results
        report_file = Path("agent2_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "tests_run": len(successful_results),
                    "average_compliance": avg_compliance,
                    "average_quality": avg_quality,
                    "average_processing_time": avg_processing_time,
                    "average_entities": avg_entities,
                    "average_relationships": avg_relationships,
                    "fully_compliant_count": fully_compliant_count,
                    "high_quality_count": high_quality_count,
                    "overall_rating": overall_rating
                },
                "detailed_results": results
            }, f, indent=2, default=str)
        
        print(f"💾 Detailed report saved to: {report_file}")


async def main():
    """Main function with command line argument handling"""
    parser = argparse.ArgumentParser(description="Agent 2 (Knowledge Extraction) Comprehensive Validation")
    parser.add_argument("--file", type=str, help="Specific file to test")
    parser.add_argument("--test-multiple", action="store_true", help="Test multiple files")
    parser.add_argument("--data-dir", type=str, default="/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output", 
                       help="Data directory for multiple file testing")
    parser.add_argument("--max-files", type=int, default=3, help="Maximum number of files to test")
    
    args = parser.parse_args()
    
    print("🧪 AGENT 2 (KNOWLEDGE EXTRACTION) COMPREHENSIVE VALIDATION")
    print("=" * 65)
    print("Purpose: Validate Agent 2 schema compliance and extraction quality")
    print("Scope: Real Azure services with complete entity/relationship analysis")
    
    validator = Agent2Validator()
    
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
        result = await validator.validate_agent2_output(content, file_path.name)
        
    else:
        # Test with default sample
        data_dir = Path("/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output")
        if data_dir.exists():
            sample_files = list(data_dir.glob("*.md"))
            if sample_files:
                sample_file = sample_files[0]
                content = sample_file.read_text(encoding='utf-8', errors='ignore')
                result = await validator.validate_agent2_output(content, sample_file.name)
            else:
                print("❌ No sample files found for testing")
        else:
            print(f"❌ Sample data directory not found: {data_dir}")
    
    # Clean up connections to prevent warnings
    try:
        from agents.core.universal_deps import cleanup_universal_deps
        await cleanup_universal_deps()
    except Exception:
        pass  # Ignore cleanup errors


if __name__ == "__main__":
    asyncio.run(main())