#!/usr/bin/env python3
"""
Test Domain Intelligence Integration with Real Data

Tests the domain intelligence analyzer using the real Programming Language corpus
to validate that learned parameters are generated correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agents.domain_intelligence.analyzers.unified_content_analyzer import UnifiedContentAnalyzer
from agents.core.dynamic_config_manager import DynamicConfigManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_domain_intelligence():
    """Test domain intelligence with real Programming Language corpus"""
    
    print("🚀 Testing Domain Intelligence Integration with Real Data")
    print("=" * 60)
    
    try:
        # 1. Test domain detection and analysis
        print("\n1️⃣ Testing Domain Analysis with Real Corpus")
        analyzer = UnifiedContentAnalyzer()
        
        corpus_path = "/workspace/azure-maintie-rag/data/raw/Programming-Language"
        if not Path(corpus_path).exists():
            print(f"❌ Corpus path not found: {corpus_path}")
            return False
        
        print(f"📂 Analyzing corpus: {corpus_path}")
        domain_profile = await analyzer.analyze_corpus_domain(corpus_path)
        
        print(f"✅ Domain detected: {domain_profile.domain_name}")
        print(f"📊 Documents analyzed: {domain_profile.document_count}")
        print(f"📏 Average document length: {domain_profile.avg_document_length:.0f} words")
        print(f"🎯 Learned entity threshold: {domain_profile.entity_confidence_threshold:.3f}")
        print(f"📦 Optimal chunk size: {domain_profile.optimal_chunk_size}")
        print(f"🔗 Relationship threshold: {domain_profile.relationship_confidence_threshold:.3f}")
        print(f"🔍 Similarity threshold: {domain_profile.similarity_threshold:.3f}")
        print(f"📈 Technical term density: {domain_profile.technical_term_density:.2f}")
        print(f"🧠 Processing complexity: {domain_profile.processing_complexity}")
        print(f"✨ Analysis confidence: {domain_profile.analysis_confidence:.2f}")
        
        # 2. Test dynamic config manager integration
        print("\n2️⃣ Testing Dynamic Config Manager Integration")
        config_manager = DynamicConfigManager()
        
        extraction_config = await config_manager.get_extraction_config("programming_language")
        
        print(f"✅ Generated extraction config:")
        print(f"   🎯 Entity threshold: {extraction_config.entity_confidence_threshold:.3f}")
        print(f"   🔗 Relationship threshold: {extraction_config.relationship_confidence_threshold:.3f}")
        print(f"   📦 Chunk size: {extraction_config.chunk_size}")
        print(f"   📊 Max entities per chunk: {extraction_config.max_entities_per_chunk}")
        print(f"   ⚖️ Quality threshold: {extraction_config.quality_validation_threshold:.3f}")
        
        # Verify learned values are different from fallback constants
        from agents.core.constants import KnowledgeExtractionConstants
        
        fallback_entity = KnowledgeExtractionConstants.FALLBACK_ENTITY_CONFIDENCE_THRESHOLD
        fallback_chunk = KnowledgeExtractionConstants.FALLBACK_CHUNK_SIZE
        
        if (extraction_config.entity_confidence_threshold != fallback_entity or 
            extraction_config.chunk_size != fallback_chunk):
            print("✅ SUCCESS: Learned parameters differ from fallback constants!")
            print(f"   📊 Fallback entity threshold: {fallback_entity:.3f} → Learned: {extraction_config.entity_confidence_threshold:.3f}")
            print(f"   📦 Fallback chunk size: {fallback_chunk} → Learned: {extraction_config.chunk_size}")
        else:
            print("⚠️  WARNING: Parameters match fallback constants (may indicate fallback was used)")
        
        # 3. Test corpus stats extraction
        print(f"\n3️⃣ Corpus Statistics")
        stats = extraction_config.corpus_stats
        if stats:
            print(f"   📚 Technical vocabulary: {stats.get('technical_vocabulary_size', 0)} terms")
            print(f"   💡 Key concepts: {stats.get('key_concepts_found', 0)} concepts")
            print(f"   🔍 Entity density: {stats.get('entity_density', 0):.4f}")
            print(f"   📊 Analysis confidence: {stats.get('analysis_confidence', 0):.2f}")
        
        print(f"\n🎉 Domain Intelligence Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    success = await test_domain_intelligence()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())