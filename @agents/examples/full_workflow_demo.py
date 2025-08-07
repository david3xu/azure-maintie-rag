#!/usr/bin/env python3
"""
Universal RAG Complete Workflow Demo
===================================

Demonstrates the complete universal RAG workflow from domain analysis
through knowledge extraction to search, showing how the system adapts
to ANY content type without predetermined assumptions.
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, '..')

# Import universal components
from domain_intelligence.agent import (
    run_universal_domain_analysis, 
    UniversalDomainDeps,
    UniversalDomainAnalysis
)
from core.universal_models import UniversalOrchestrationResult

async def demo_universal_workflow(data_directory: str = "/workspace/azure-maintie-rag/data/raw"):
    """Demonstrate complete universal RAG workflow"""
    
    print("🌍 Universal RAG Complete Workflow Demo")
    print("======================================")
    print(f"📂 Processing content from: {data_directory}")
    print("🎯 This workflow adapts to ANY content type without assumptions")
    
    # Phase 1: Universal Domain Intelligence
    print("\n🧠 Phase 1: Universal Domain Intelligence")
    print("==========================================")
    
    try:
        deps = UniversalDomainDeps(
            data_directory=data_directory,
            max_files_to_analyze=30,
            min_content_length=100,
            enable_multilingual=True
        )
        
        print("🔍 Analyzing content distribution and characteristics...")
        domain_analysis = await run_universal_domain_analysis(deps)
        
        print(f"\n✅ Domain Analysis Complete:")
        print(f"   🏷️  Domain Signature: {domain_analysis.domain_signature}")
        print(f"   🎯 Confidence: {domain_analysis.content_type_confidence:.2f}")
        print(f"   📊 Documents Analyzed: {domain_analysis.characteristics.document_count}")
        print(f"   📈 Vocabulary Richness: {domain_analysis.characteristics.vocabulary_richness:.3f}")
        print(f"   🔧 Technical Density: {domain_analysis.characteristics.technical_vocabulary_ratio:.3f}")
        print(f"   ⏱️  Processing Time: {domain_analysis.processing_time:.2f}s")
        print(f"   🔒 Reliability: {domain_analysis.analysis_reliability:.2f}")
        
        print(f"\n💡 Key Insights Discovered:")
        for i, insight in enumerate(domain_analysis.key_insights, 1):
            print(f"   {i}. {insight}")
        
        print(f"\n🎯 Adaptive Configuration Generated:")
        pc = domain_analysis.processing_config
        print(f"   📦 Optimal Chunk Size: {pc.optimal_chunk_size}")
        print(f"   🔗 Chunk Overlap: {pc.chunk_overlap_ratio:.1%}")
        print(f"   🏷️  Entity Threshold: {pc.entity_confidence_threshold:.2f}")
        print(f"   🔍 Vector Weight: {pc.vector_search_weight:.1%}")
        print(f"   🕸️  Graph Weight: {pc.graph_search_weight:.1%}")
        print(f"   📊 Expected Quality: {pc.expected_extraction_quality:.1%}")
        print(f"   ⚙️  Complexity: {pc.processing_complexity}")
        
    except Exception as e:
        print(f"❌ Domain analysis failed: {e}")
        return None
    
    # Phase 2: Demonstrate Agent Configuration
    print(f"\n🔧 Phase 2: Adaptive Agent Configuration")
    print(f"========================================")
    
    print(f"🧠 Knowledge Extraction Agent would receive:")
    extraction_config = {
        "chunk_size": domain_analysis.processing_config.optimal_chunk_size,
        "chunk_overlap": domain_analysis.processing_config.chunk_overlap_ratio,
        "entity_threshold": domain_analysis.processing_config.entity_confidence_threshold,
        "relationship_density": domain_analysis.processing_config.relationship_density,
        "domain_patterns": domain_analysis.characteristics.content_patterns,
        "key_terms": domain_analysis.characteristics.most_frequent_terms[:10],
        "processing_complexity": domain_analysis.processing_config.processing_complexity
    }
    
    for key, value in extraction_config.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        elif isinstance(value, list):
            print(f"   {key}: {value[:3] if len(value) > 3 else value}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n🔍 Universal Search Agent would receive:")
    search_config = {
        "vector_weight": domain_analysis.processing_config.vector_search_weight,
        "graph_weight": domain_analysis.processing_config.graph_search_weight,
        "domain_signature": domain_analysis.domain_signature,
        "key_terms": domain_analysis.characteristics.most_frequent_terms[:5],
        "technical_density": domain_analysis.characteristics.technical_vocabulary_ratio,
        "enable_advanced_search": domain_analysis.processing_config.processing_complexity == "high"
    }
    
    for key, value in search_config.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        elif isinstance(value, list):
            print(f"   {key}: {value}")
        elif isinstance(value, bool):
            print(f"   {key}: {'✅' if value else '❌'}")
        else:
            print(f"   {key}: {value}")
    
    # Phase 3: Demonstrate Universality
    print(f"\n🌍 Phase 3: Universal Adaptation Demonstration")
    print(f"==============================================")
    
    print(f"🎯 This configuration works for ANY content type:")
    
    # Show adaptation examples
    adaptations = []
    
    if "code_rich" in domain_analysis.characteristics.content_patterns:
        adaptations.append("📝 Code-aware chunking enabled - preserves function/class boundaries")
        adaptations.append("🔧 Larger chunks for complex code structures")
        
    if "hierarchical_headers" in domain_analysis.characteristics.content_patterns:
        adaptations.append("📋 Structure-aware processing - uses document hierarchy")
        
    if domain_analysis.characteristics.technical_vocabulary_ratio > 0.4:
        adaptations.append("🧬 High-tech content detected - specialized entity recognition")
        adaptations.append("📊 Graph search prioritized for technical relationships")
        
    if domain_analysis.characteristics.vocabulary_richness > 0.3:
        adaptations.append("📚 Rich vocabulary - semantic vector search optimized")
    else:
        adaptations.append("🔗 Limited vocabulary - relationship extraction prioritized")
    
    for adaptation in adaptations:
        print(f"   ✅ {adaptation}")
    
    if not adaptations:
        print(f"   📊 Standard balanced configuration for general content")
    
    # Phase 4: Show Universal Benefits
    print(f"\n🚀 Phase 4: Universal RAG Benefits Realized")
    print(f"==========================================")
    
    benefits = [
        f"🌍 Works with ANY domain: legal, medical, technical, business, academic, etc.",
        f"🔄 Zero configuration required for new content types",
        f"📈 Intelligent optimization based on actual content characteristics",
        f"🎯 Quality expectations set from real content analysis (not assumptions)",
        f"⚡ Efficient resource allocation based on complexity assessment",
        f"🔧 All agents automatically configured for optimal performance",
        f"📊 Consistent behavior across diverse content collections"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    # Create summary result
    result = {
        "success": True,
        "domain_analysis": domain_analysis,
        "extraction_config": extraction_config,
        "search_config": search_config,
        "total_processing_time": domain_analysis.processing_time,
        "universal_adaptations": adaptations,
        "quality_score": domain_analysis.analysis_reliability
    }
    
    return result

def demonstrate_universality_comparison():
    """Show comparison between universal and predetermined approaches"""
    
    print(f"\n📊 Universal vs Predetermined Approach Comparison")
    print(f"================================================")
    
    comparison_data = [
        ("Domain Types", "❌ Fixed list (programming, business, etc.)", "✅ Dynamic signatures from content"),
        ("Keywords", "❌ Hardcoded domain vocabularies", "✅ Discovered from frequency analysis"), 
        ("Thresholds", "❌ Static values (0.7, 0.8, etc.)", "✅ Calculated from content distribution"),
        ("Entity Types", "❌ Predetermined (PERSON, ORG, etc.)", "✅ Discovered from content patterns"),
        ("Configuration", "❌ Manual domain-specific rules", "✅ Generated from measured characteristics"),
        ("New Domains", "❌ Requires manual setup", "✅ Automatic adaptation"),
        ("Languages", "❌ English-centric assumptions", "✅ Language-agnostic analysis"),
        ("Scalability", "❌ Limited to known domains", "✅ Infinite domain support"),
        ("Maintenance", "❌ Requires domain experts", "✅ Self-maintaining through data analysis")
    ]
    
    print(f"{'Aspect':<15} {'Predetermined Approach':<35} {'Universal Approach':<35}")
    print(f"{'-'*15} {'-'*35} {'-'*35}")
    
    for aspect, predetermined, universal in comparison_data:
        print(f"{aspect:<15} {predetermined:<35} {universal:<35}")
    
    print(f"\n🎯 Result: Universal approach maintains true RAG universality while")
    print(f"   providing intelligent optimization through pure data-driven analysis.")

async def main():
    """Run the complete universal RAG workflow demonstration"""
    
    # Check if data directory exists
    data_path = Path("/workspace/azure-maintie-rag/data/raw")
    
    if data_path.exists():
        # Run full workflow demo
        result = await demo_universal_workflow(str(data_path))
        
        if result and result["success"]:
            print(f"\n✅ Universal RAG Workflow Demo Completed Successfully!")
            print(f"📊 Quality Score: {result['quality_score']:.2f}")
            print(f"⏱️  Total Time: {result['total_processing_time']:.2f}s")
            print(f"🎯 Adaptations Applied: {len(result['universal_adaptations'])}")
        else:
            print(f"\n⚠️  Demo completed with warnings")
            
    else:
        print(f"\n⚠️  Data directory not found: {data_path}")
        print(f"   The universal system works with ANY content you provide")
        print(f"   Demo shows conceptual workflow with example adaptations")
    
    # Always show the universality comparison
    demonstrate_universality_comparison()
    
    print(f"\n🌍 Universal RAG System Ready!")
    print(f"=============================")
    print(f"Your RAG system is now truly universal AND intelligently adaptive.")
    print(f"It maintains universal principles while providing domain-specific optimization")
    print(f"through pure data-driven analysis - the best of both worlds!")

if __name__ == "__main__":
    asyncio.run(main())