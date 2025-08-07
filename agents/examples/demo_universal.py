#!/usr/bin/env python3
"""
Universal Domain Intelligence Demo - PydanticAI Architecture
===========================================================

Demonstrates the Domain Intelligence Agent using proper PydanticAI patterns:
- Universal content analysis without hardcoded assumptions
- Proper dependency injection and service management
- Real Azure OpenAI integration for content discovery
"""

import asyncio
from pathlib import Path

# Import PydanticAI Domain Intelligence Agent
from agents.domain_intelligence.agent import run_domain_analysis
from agents.core.universal_deps import get_universal_deps


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


async def run_pydantic_ai_domain_demo():
    """Run the PydanticAI Domain Intelligence Agent demo"""

    # Sample content to analyze (since we're focusing on the PydanticAI architecture)
    sample_content = """
    Machine Learning Operations (MLOps) is the practice of applying DevOps principles to machine learning
    workflows. It encompasses the entire ML lifecycle including data preparation, model training, 
    deployment, monitoring, and maintenance. Key components include automated pipelines, version control
    for datasets and models, continuous integration/continuous deployment (CI/CD) for ML, model 
    performance monitoring, and automated retraining strategies.
    
    MLOps frameworks like Kubeflow, MLflow, and Azure ML provide infrastructure for managing ML 
    experiments, tracking model metrics, and orchestrating complex workflows. Feature stores enable
    consistent feature engineering across training and serving environments. Model registries 
    maintain versioned models with lineage tracking and governance policies.
    """

    print(f"\n🚀 Running PydanticAI Domain Intelligence Agent")
    print(f"==============================================")
    print(
        f"🎯 Demonstrating proper agent architecture with universal content processing"
    )
    print(f"📊 Using sample content to show characteristic discovery")

    try:
        # Initialize dependencies properly
        deps = await get_universal_deps()
        print(
            f"✅ UniversalDeps initialized with services: {', '.join(deps.get_available_services())}"
        )

        # Run the PydanticAI Domain Intelligence Agent
        print(f"\n🧠 Running Domain Intelligence Agent...")
        analysis = await run_domain_analysis(sample_content, detailed=True)

        print(f"\n📈 PydanticAI Agent Results:")
        print(f"============================")
        print(f"🏷️  Content Signature: {analysis.content_signature}")
        print(f"📊 Vocabulary Complexity: {analysis.vocabulary_complexity:.3f}")
        print(f"🎯 Concept Density: {analysis.concept_density:.3f}")
        print(f"🔍 Discovered Patterns: {analysis.discovered_patterns}")

        if hasattr(analysis, "entity_indicators"):
            print(f"🏷️  Entity Indicators: {analysis.entity_indicators}")
        if hasattr(analysis, "relationship_indicators"):
            print(f"🔗 Relationship Indicators: {analysis.relationship_indicators}")

        print(f"\n🔧 Agent Architecture Benefits:")
        print(f"==============================")
        print(
            f"✅ **Proper PydanticAI Patterns**: Agent uses RunContext[UniversalDeps]"
        )
        print(f"✅ **Centralized Dependencies**: Shared Azure services, no duplication")
        print(f"✅ **Universal Processing**: Zero hardcoded domain assumptions")
        print(f"✅ **Atomic Tools**: Clean tool boundaries for content analysis")
        print(f"✅ **Real Azure Integration**: Uses actual Azure OpenAI service")
        print(f"✅ **Factory Functions**: Proper agent initialization patterns")

        print(f"\n🎛️ How This Enables Multi-Agent Coordination:")
        print(f"=============================================")
        print(
            f"🔗 **Agent Delegation**: Other agents can call this with ctx.deps/ctx.usage"
        )
        print(
            f"📊 **Shared Configuration**: Analysis informs Knowledge Extraction Agent"
        )
        print(
            f"🔍 **Search Optimization**: Characteristics optimize Universal Search Agent"
        )
        print(f"⚙️  **Orchestration**: Results drive workflow decision-making")

        return analysis

    except Exception as e:
        print(f"❌ PydanticAI agent demo failed: {e}")
        print(f"This may be due to Azure service configuration issues")
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
    """Run the complete PydanticAI Domain Intelligence Agent demo"""

    # Show the universality principles
    demonstrate_universality()

    # Run the PydanticAI agent demonstration
    try:
        analysis = await run_pydantic_ai_domain_demo()
        if analysis:
            print(f"\n✅ PydanticAI Domain Intelligence demo completed successfully!")
            print(f"🏷️  Generated signature: {analysis.content_signature}")
        else:
            print(f"\n⚠️  Demo completed with warnings (likely Azure service issues)")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("This may be due to Azure service configuration or connectivity issues")

    # Show comparison with predetermined approaches
    compare_approaches()

    print(f"\n🌟 PydanticAI Universal RAG Benefits")
    print(f"===================================")
    print(f"✅ **Architecture**: Proper multi-agent patterns with dependency injection")
    print(f"✅ **Universality**: Zero hardcoded domain assumptions preserved")
    print(f"✅ **Integration**: Real Azure services (OpenAI, Cosmos DB, Search)")
    print(f"✅ **Coordination**: Agents delegate to each other with ctx.deps/ctx.usage")
    print(f"✅ **Scalability**: Clean boundaries enable easy testing and extension")
    print(f"")
    print(f"🎯 Your system is now truly universal AND properly architected!")


if __name__ == "__main__":
    asyncio.run(main())
