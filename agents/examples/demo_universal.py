#!/usr/bin/env python3
"""
Universal Domain Intelligence Demo - Zero Hardcoded Values
=========================================================

Demonstrates how the universal agent discovers domain characteristics
from actual data without any predetermined assumptions.
"""

import sys
import asyncio
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import universal domain intelligence
from agents.domain_intelligence.agent import (
    UniversalDomainDeps, 
    run_universal_domain_analysis
)

def demonstrate_universality():
    """Show how the universal agent has ZERO hardcoded domain knowledge"""
    
    print("🌍 Universal Domain Intelligence - Zero Hardcoded Values")
    print("=====================================================")
    
    print("\n✅ What makes this TRULY universal:")
    print("   • NO predetermined domain types (programming, business, etc.)")
    print("   • NO hardcoded keywords or entity types") 
    print("   • NO fixed thresholds or scoring rules")
    print("   • NO language assumptions")
    print("   • NO content structure assumptions")
    
    print("\n🔍 Instead, it discovers:")
    print("   • Domain characteristics from vocabulary patterns")
    print("   • Content structure from actual document analysis")
    print("   • Processing parameters from measured complexity")
    print("   • Thresholds from content distribution statistics")
    print("   • Configuration from discovered patterns")
    
    print("\n📊 Data-Driven Analysis Process:")
    print("   1. Statistical analysis of actual content")
    print("   2. Vocabulary richness and technical density measurement")
    print("   3. Structural pattern discovery")
    print("   4. Adaptive configuration generation")
    print("   5. Dynamic signature creation from characteristics")

async def run_universal_demo():
    """Run the universal analysis on real data"""
    
    # Test with actual data directory
    deps = UniversalDomainDeps(
        data_directory="/workspace/azure-maintie-rag/data/raw",
        max_files_to_analyze=30,  # Reasonable limit
        min_content_length=50,
        enable_multilingual=True
    )
    
    print(f"\n🚀 Running Universal Analysis on Real Data")
    print(f"==========================================")
    print(f"📂 Data source: {deps.data_directory}")
    print(f"📊 Analysis limit: {deps.max_files_to_analyze} files")
    print(f"🌐 Multilingual support: {'✅' if deps.enable_multilingual else '❌'}")
    
    # Run the universal analysis
    try:
        analysis = await run_universal_domain_analysis(deps)
        
        print(f"\n📈 Universal Analysis Results:")
        print(f"===============================")
        print(f"🏷️  Domain Signature: {analysis.domain_signature}")
        print(f"🎯 Confidence: {analysis.content_type_confidence:.2f}")
        print(f"⏱️  Processing Time: {analysis.processing_time:.2f}s")
        print(f"🔒 Reliability: {analysis.analysis_reliability:.2f}")
        
        print(f"\n📊 Discovered Content Characteristics:")
        print(f"======================================")
        cc = analysis.characteristics
        print(f"   Documents Analyzed: {cc.document_count}")
        print(f"   Avg Document Length: {cc.avg_document_length:,} chars")
        print(f"   Vocabulary Richness: {cc.vocabulary_richness:.3f}")
        print(f"   Technical Density: {cc.technical_vocabulary_ratio:.3f}")
        print(f"   Sentence Complexity: {cc.sentence_complexity:.1f} words/sentence")
        print(f"   Lexical Diversity: {cc.lexical_diversity:.3f}")
        
        if cc.most_frequent_terms:
            print(f"   Most Frequent Terms: {cc.most_frequent_terms[:5]}")
        
        if cc.content_patterns:
            print(f"   Content Patterns: {cc.content_patterns}")
        
        print(f"\n⚙️  Adaptive Processing Configuration:")
        print(f"=====================================")
        pc = analysis.processing_config
        print(f"   Optimal Chunk Size: {pc.optimal_chunk_size}")
        print(f"   Chunk Overlap: {pc.chunk_overlap_ratio:.1%}")
        print(f"   Entity Threshold: {pc.entity_confidence_threshold:.2f}")
        print(f"   Vector Search Weight: {pc.vector_search_weight:.1%}")
        print(f"   Graph Search Weight: {pc.graph_search_weight:.1%}")
        print(f"   Expected Quality: {pc.expected_extraction_quality:.1%}")
        print(f"   Processing Complexity: {pc.processing_complexity}")
        
        print(f"\n💡 Data-Driven Insights:")
        print(f"========================")
        for i, insight in enumerate(analysis.key_insights, 1):
            print(f"   {i}. {insight}")
        
        print(f"\n🎯 Adaptation Recommendations:")
        print(f"==============================")
        for i, rec in enumerate(analysis.adaptation_recommendations, 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🔗 How This Drives Other Agents:")
        print(f"=================================")
        print(f"   🧠 Knowledge Extraction Agent gets:")
        print(f"      • Chunk size: {pc.optimal_chunk_size}")
        print(f"      • Entity threshold: {pc.entity_confidence_threshold}")
        print(f"      • Content patterns: {cc.content_patterns}")
        
        print(f"   🔍 Universal Search Agent gets:")
        print(f"      • Vector weight: {pc.vector_search_weight}")
        print(f"      • Graph weight: {pc.graph_search_weight}")
        print(f"      • Domain terms: {cc.most_frequent_terms[:3] if cc.most_frequent_terms else 'None'}")
        
        print(f"   🎛️  Orchestrator gets:")
        print(f"      • Processing complexity: {pc.processing_complexity}")
        print(f"      • Expected quality: {pc.expected_extraction_quality}")
        print(f"      • Reliability score: {analysis.analysis_reliability}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def compare_approaches():
    """Compare universal vs predetermined approaches"""
    
    print(f"\n📊 Universal vs Predetermined Comparison")
    print(f"=======================================")
    
    print(f"📉 Predetermined Approach Problems:")
    print(f"   ❌ Fixed domain types (programming, business, etc.)")
    print(f"   ❌ Hardcoded keywords and patterns")
    print(f"   ❌ Static configuration values")
    print(f"   ❌ Language-specific assumptions")
    print(f"   ❌ Fails on unknown/mixed domains")
    print(f"   ❌ Not adaptable to new content types")
    
    print(f"\n📈 Universal Approach Benefits:")
    print(f"   ✅ Learns from actual content characteristics")
    print(f"   ✅ Adapts to ANY domain or content type")
    print(f"   ✅ No hardcoded assumptions")
    print(f"   ✅ Language-agnostic analysis")
    print(f"   ✅ Handles mixed/unknown domains gracefully")
    print(f"   ✅ Generates optimal configurations dynamically")
    print(f"   ✅ Maintains true 'universal' RAG principles")
    
    print(f"\n🎯 Result:")
    print(f"   The universal agent preserves the 'universal' nature of your RAG")
    print(f"   system while still providing intelligent domain-specific optimization.")

async def main():
    """Run the complete universal domain intelligence demo"""
    
    # Show the universality principles
    demonstrate_universality()
    
    # Run actual analysis if data directory exists
    data_path = Path("/workspace/azure-maintie-rag/data/raw")
    if data_path.exists():
        analysis = await run_universal_demo()
        if analysis:
            print(f"\n✅ Universal analysis completed successfully!")
        else:
            print(f"\n⚠️  Analysis completed with warnings")
    else:
        print(f"\n⚠️  Data directory not found: {data_path}")
        print(f"   The universal agent will analyze any content you provide")
    
    # Show comparison with predetermined approaches
    compare_approaches()
    
    print(f"\n🌍 Universal RAG Principles Maintained!")
    print(f"======================================")
    print(f"Your system remains truly universal while gaining intelligent")
    print(f"domain-specific optimization through data-driven analysis.")

if __name__ == "__main__":
    asyncio.run(main())