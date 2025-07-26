#!/usr/bin/env python3
"""
Test Context-Aware Real Extraction with Azure OpenAI
Uses real maintenance data and actual Azure OpenAI API calls
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.azure_openai.improved_extraction_client import ImprovedKnowledgeExtractor

def load_real_maintenance_texts(limit: int = 10) -> list[str]:
    """Load real maintenance texts from the dataset"""
    
    raw_file = Path(__file__).parent.parent / "data" / "raw" / "maintenance_all_texts.md"
    
    if not raw_file.exists():
        print(f"❌ Raw maintenance file not found: {raw_file}")
        return [
            "air conditioner thermostat not working",
            "bearing on air conditioner compressor unserviceable", 
            "blown o-ring off steering hose",
            "brake system pressure low",
            "coolant temperature sensor malfunction"
        ]
    
    texts = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('<id>'):
                text_content = line.replace('<id>', '').strip()
                if text_content:
                    texts.append(text_content)
                    if len(texts) >= limit:
                        break
    
    print(f"✅ Loaded {len(texts)} real maintenance texts")
    return texts

def main():
    """Test context-aware extraction with real data"""
    
    print("🧪 Testing Context-Aware Real Extraction")
    print("=" * 60)
    
    # Load real maintenance texts
    texts = load_real_maintenance_texts(limit=5)  # Start with 5 for testing
    
    print(f"📝 Processing {len(texts)} real maintenance texts:")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # Initialize improved extractor with context-aware templates
    try:
        extractor = ImprovedKnowledgeExtractor("maintenance")
        print("✅ Extractor initialized with context-aware templates")
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        return
    
    # Run extraction with comparison
    try:
        print(f"\n🔍 Running context-aware extraction...")
        results = extractor.extract_with_comparison(texts, sample_size=len(texts))
        
        print(f"\n📊 EXTRACTION SUMMARY:")
        summary = results.get('summary', {})
        print(f"   • Texts processed: {summary.get('total_texts_processed', 0)}")
        print(f"   • Total entities: {summary.get('total_entities_extracted', 0)}")
        print(f"   • Total relationships: {summary.get('total_relations_extracted', 0)}")
        print(f"   • Avg entities per text: {summary.get('avg_entities_per_text', 0)}")
        print(f"   • Avg relationships per text: {summary.get('avg_relations_per_text', 0)}")
        
        quality_metrics = summary.get('quality_metrics', {})
        print(f"\n📈 QUALITY METRICS:")
        print(f"   • Overall quality: {quality_metrics.get('avg_overall_quality', 0):.3f}")
        print(f"   • Context preservation: {quality_metrics.get('avg_context_preservation', 0):.3f}")
        print(f"   • Entity coverage: {quality_metrics.get('avg_entity_coverage', 0):.3f}")
        print(f"   • Connectivity: {quality_metrics.get('avg_connectivity', 0):.3f}")
        
        common_issues = summary.get('common_issues', {})
        if common_issues:
            print(f"\n⚠️  COMMON ISSUES:")
            for issue, count in common_issues.items():
                print(f"   • {issue}: {count} texts")
        
        print(f"\n✅ Context-aware extraction completed successfully!")
        print(f"📄 Detailed results saved in data/extraction_comparisons/")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()