#!/usr/bin/env python3
"""
Debug Agent 1 Output Issues
============================

Direct analysis of Domain Intelligence Agent output to identify specific problems
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.domain_intelligence.agent import run_domain_analysis
from agents.core.universal_deps import get_universal_deps


async def debug_agent1_output():
    """Debug Agent 1 output issues with various content types"""
    
    print("🔍 Agent 1 (Domain Intelligence) Output Analysis")
    print("=" * 60)
    
    # Test cases representing different content complexity
    test_cases = [
        {
            "name": "Simple Technical Content",
            "content": """
            Azure OpenAI Service provides REST API access to OpenAI's powerful language models including GPT-4, GPT-3.5-turbo, and Codex series.
            The service offers the same capabilities as OpenAI GPT models with the security and enterprise-grade features of Azure.
            """
        },
        {
            "name": "Complex Documentation",  
            "content": """
            The Azure Cognitive Services Language Understanding (LUIS) application uses machine learning to understand natural language text.
            LUIS applications can be configured with intents, entities, and utterances to build sophisticated conversational AI solutions.
            Integration with Bot Framework enables multi-channel deployment across Teams, Slack, and web interfaces.
            Key features include:
            - Intent recognition with confidence scoring
            - Entity extraction with pre-built and custom models
            - Multi-language support with automatic detection
            - Batch testing and model versioning capabilities
            - Active learning for continuous improvement
            """
        },
        {
            "name": "Code-Heavy Content",
            "content": """
            ```python
            import azure.cognitiveservices.language.luis.runtime as luis_runtime
            from azure.cognitiveservices.language.luis.runtime.models import LuisResult
            
            def predict_intent(app_id: str, query: str) -> LuisResult:
                luis_client = luis_runtime.LUISRuntimeClient(endpoint, credentials)
                prediction_request = PredictionRequest(query=query)
                return luis_client.prediction.get_slot_prediction(
                    app_id=app_id,
                    slot_name="production", 
                    prediction_request=prediction_request
                )
            ```
            
            The LUIS runtime API provides real-time prediction capabilities with the following parameters:
            - app_id: Unique identifier for the LUIS application
            - slot_name: Deployment slot (production/staging)
            - prediction_request: Query and optional parameters
            """
        }
    ]
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Run domain analysis
            result = await run_domain_analysis(test_case['content'], detailed=True)
            
            # Analyze the output structure and quality
            output_analysis = {
                "domain_signature": result.domain_signature,
                "characteristics": {
                    "vocabulary_complexity": result.characteristics.vocabulary_complexity_ratio,
                    "concept_density": result.characteristics.concept_density,
                    "lexical_diversity": result.characteristics.lexical_diversity,
                    "structural_patterns": result.characteristics.structural_patterns,
                },
                "processing_config": {
                    "chunk_size": result.processing_config.optimal_chunk_size,
                    "overlap_ratio": result.processing_config.chunk_overlap_ratio,
                    "entity_threshold": result.processing_config.entity_confidence_threshold,
                    "complexity": result.processing_config.processing_complexity,
                },
                "issues_detected": []
            }
            
            # Detect potential output issues
            issues = []
            
            # Check for unrealistic values
            if result.characteristics.vocabulary_complexity_ratio >= 1.0:
                issues.append("Vocabulary complexity at maximum (1.0) - may indicate calculation error")
            
            if result.characteristics.concept_density >= 1.0:
                issues.append("Concept density at maximum (1.0) - may indicate calculation error")
                
            # Check for missing or empty patterns
            if not result.characteristics.structural_patterns:
                issues.append("No structural patterns detected - analysis may be incomplete")
                
            # Check signature consistency
            signature_parts = result.domain_signature.split('_')
            if len(signature_parts) != 3:
                issues.append(f"Domain signature format inconsistent: {result.domain_signature}")
                
            # Check processing config reasonableness
            if result.processing_config.optimal_chunk_size < 100 or result.processing_config.optimal_chunk_size > 4000:
                issues.append(f"Unusual chunk size: {result.processing_config.optimal_chunk_size}")
                
            if result.processing_config.chunk_overlap_ratio > 0.5:
                issues.append(f"High overlap ratio: {result.processing_config.chunk_overlap_ratio}")
            
            output_analysis["issues_detected"] = issues
            results[test_case['name']] = output_analysis
            
            # Print summary
            print(f"✅ Domain Signature: {result.domain_signature}")
            print(f"📊 Vocabulary Complexity: {result.characteristics.vocabulary_complexity_ratio:.3f}")
            print(f"🧠 Concept Density: {result.characteristics.concept_density:.3f}")
            print(f"🏗️  Structural Patterns: {len(result.characteristics.structural_patterns)} found")
            print(f"⚙️  Chunk Size: {result.processing_config.optimal_chunk_size}")
            print(f"🔧 Processing Complexity: {result.processing_config.processing_complexity}")
            
            if issues:
                print(f"⚠️  Issues Detected ({len(issues)}):")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ No obvious issues detected")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results[test_case['name']] = {"error": str(e)}
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY OF AGENT 1 OUTPUT ISSUES")
    print("=" * 60)
    
    all_issues = []
    successful_tests = 0
    
    for test_name, result in results.items():
        if "error" not in result:
            successful_tests += 1
            if result.get("issues_detected"):
                all_issues.extend([f"{test_name}: {issue}" for issue in result["issues_detected"]])
    
    print(f"✅ Successful Analyses: {successful_tests}/{len(test_cases)}")
    print(f"⚠️  Total Issues Found: {len(all_issues)}")
    
    if all_issues:
        print("\nDETAILED ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
    
    # Save detailed results
    with open('/workspace/azure-maintie-rag/agent1_debug_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: /workspace/azure-maintie-rag/agent1_debug_results.json")
    
    return results


if __name__ == "__main__":
    asyncio.run(debug_agent1_output())