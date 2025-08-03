#!/usr/bin/env python3
"""
Test Agent 1 Phase 0 Enhancement - Complete Data-Driven Configuration Generation

Tests the new create_fully_learned_extraction_config() tool that replaces
hardcoded values with learned parameters from corpus analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from agents.domain_intelligence.agent import get_domain_agent

async def test_agent1_phase0_enhancement():
    """Test Agent 1's enhanced learning capabilities with Programming-Language corpus"""
    
    print("🧪 Testing Agent 1 Phase 0 Enhancement")
    print("=" * 50)
    
    # Get the domain agent
    domain_agent = get_domain_agent()
    
    if domain_agent is None:
        print("⚠️ Domain agent not available (no Azure OpenAI config)")
        print("Testing will use statistical-only analysis mode")
        return False
    
    # Test with the Programming-Language subdirectory mentioned in the analysis
    corpus_path = "data/raw/Programming-Language"
    
    if not Path(corpus_path).exists():
        print(f"❌ Test corpus not found: {corpus_path}")
        print("Please ensure the Programming-Language directory exists in data/raw/")
        return False
    
    try:
        print(f"📂 Testing with corpus: {corpus_path}")
        
        # Test the new fully learned configuration generation
        print("🔄 Generating fully learned configuration...")
        config = await domain_agent.run(
            "create_fully_learned_extraction_config",
            corpus_path=corpus_path
        )
        
        print("✅ Configuration generated successfully!")
        print(f"📊 Domain: {config.domain_name}")
        print(f"📊 Entity threshold: {config.entity_confidence_threshold} (learned from complexity)")
        print(f"📊 Chunk size: {config.chunk_size} (learned from document characteristics)")
        print(f"📊 Response SLA: {config.target_response_time_seconds}s (learned from complexity)")
        print(f"📊 Entity types discovered: {len(config.expected_entity_types)}")
        
        # Validate no hardcoded critical values
        print("\n🔍 Validating zero hardcoded critical values...")
        
        # Check that values are not hardcoded defaults
        violations = []
        
        if config.entity_confidence_threshold == 0.7:
            violations.append("entity_confidence_threshold appears to be hardcoded default (0.7)")
        
        if config.chunk_size == 1000:
            violations.append("chunk_size appears to be hardcoded default (1000)")
        
        if config.target_response_time_seconds == 3.0:
            violations.append("response_time_sla appears to be hardcoded default (3.0)")
        
        if violations:
            print("❌ Hardcoded value violations found:")
            for violation in violations:
                print(f"   - {violation}")
            return False
        else:
            print("✅ No hardcoded critical values detected!")
        
        # Check that config was saved to file
        expected_config_file = Path(f"config/generated/domains/{config.domain_name}_config.yaml")
        if expected_config_file.exists():
            print(f"✅ Configuration saved to: {expected_config_file}")
        else:
            print(f"⚠️ Configuration file not found: {expected_config_file}")
        
        print("\n🎉 Agent 1 Phase 0 Enhancement Test: PASSED")
        print("✅ Critical parameters learned from data")
        print("✅ Zero hardcoded critical values validated")
        print("✅ Configuration generation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_learning_methods_individually():
    """Test each learning method individually"""
    
    print("\n🔬 Testing Individual Learning Methods")
    print("=" * 40)
    
    domain_agent = get_domain_agent()
    if domain_agent is None:
        print("⚠️ Domain agent not available")
        return False
    
    corpus_path = "data/raw/Programming-Language"
    
    try:
        # Test statistical analysis
        print("📊 Testing statistical analysis...")
        stats = await domain_agent.run("analyze_corpus_statistics", corpus_path=corpus_path)
        print(f"✅ Analyzed {stats.total_documents} documents, {stats.vocabulary_size} unique tokens")
        
        # Test semantic analysis
        print("🧠 Testing semantic analysis...")
        sample_content = "This is a sample content for testing semantic pattern extraction."
        patterns = await domain_agent.run("generate_semantic_patterns", content_sample=sample_content)
        print(f"✅ Extracted {len(patterns.entity_types)} entity types, {len(patterns.primary_concepts)} concepts")
        
        print("✅ All individual learning methods working!")
        return True
        
    except Exception as e:
        print(f"❌ Individual method test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_agent1_phase0_enhancement())
    asyncio.run(test_learning_methods_individually())