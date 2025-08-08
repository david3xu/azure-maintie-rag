#!/usr/bin/env python3
"""
Universal RAG: Prompt Adaptation Comparison
==========================================

This script shows how the SAME universal template adapts to different domains
by comparing the generated prompts side-by-side.
"""

import sys
from pathlib import Path


def compare_generated_prompts():
    """Compare how universal templates adapt to different domains"""

    generated_dir = Path(
        "/workspace/azure-maintie-rag/infrastructure/prompt_workflows/generated"
    )

    print("🔍 Universal RAG: Prompt Adaptation Comparison")
    print("=" * 60)
    print()
    print(
        "Showing how the SAME universal template dynamically adapts to different content:"
    )
    print()

    # Compare entity extraction prompts
    print("📊 ENTITY EXTRACTION PROMPT ADAPTATION")
    print("-" * 50)

    domains = ["programming_tutorial", "medical_research", "maintenance_procedures"]

    for domain in domains:
        entity_file = generated_dir / f"{domain}_entity_extraction.jinja2"
        if entity_file.exists():
            content = entity_file.read_text()

            print(f"\n🎯 {domain.replace('_', ' ').title()} Domain:")
            print(f"   📄 File: {entity_file.name}")

            # Extract key adaptive elements
            lines = content.split("\n")

            # Domain signature
            for line in lines:
                if "Configuration:" in line:
                    print(f"   🔧 {line.strip()}")
                    break

            # Domain description
            for line in lines:
                if "deep understanding of" in line:
                    domain_desc = line.split("deep understanding of")[1].split(".")[0]
                    print(f"   📋 Domain Description: {domain_desc.strip()}")
                    break

            # Discovered patterns
            pattern_section = False
            patterns = []
            for line in lines:
                if "Domain Intelligence" in line:
                    pattern_section = True
                    continue
                if pattern_section and line.startswith("- **"):
                    pattern_name = line.split("**")[1]
                    patterns.append(pattern_name)
                elif pattern_section and line.startswith("**Quality"):
                    break

            if patterns:
                print(f"   🧠 Discovered Patterns: {', '.join(patterns[:3])}...")

            # Entity types
            for line in lines:
                if "Use types discovered from content analysis:" in line:
                    entity_types = line.split("analysis:")[1].strip()
                    print(f"   🏷️  Entity Types: {entity_types}")
                    break

            # Confidence thresholds
            for line in lines:
                if "threshold:" in line and "confidence" in line:
                    threshold = line.split("threshold:")[1].split(")")[0].strip()
                    print(f"   📈 Confidence Threshold: {threshold}")
                    break

    print("\n" + "=" * 60)
    print("📈 RELATIONSHIP EXTRACTION PROMPT ADAPTATION")
    print("-" * 50)

    for domain in domains:
        relation_file = generated_dir / f"{domain}_relation_extraction.jinja2"
        if relation_file.exists():
            content = relation_file.read_text()

            print(f"\n🎯 {domain.replace('_', ' ').title()} Domain:")
            print(f"   📄 File: {relation_file.name}")

            lines = content.split("\n")

            # Configuration
            for line in lines:
                if "Configuration:" in line:
                    print(f"   🔧 {line.strip()}")
                    break

            # Relationship patterns
            in_relationship_section = False
            relationship_types = []
            current_category = None

            for line in lines:
                if "Relationships**:" in line:
                    current_category = line.split("**")[1].replace(" Relationships", "")
                    in_relationship_section = True
                    continue
                elif in_relationship_section and line.startswith("- `"):
                    rel_type = line.split("`")[1]
                    if current_category:
                        relationship_types.append(f"{current_category}:{rel_type}")
                    else:
                        relationship_types.append(rel_type)
                elif in_relationship_section and (
                    line.startswith("## ") or line.startswith("**")
                ):
                    break

            if relationship_types:
                print(
                    f"   🔗 Relationship Types: {', '.join(relationship_types[:4])}..."
                )

            # Domain-specific insights
            insights = []
            in_insights = False
            for line in lines:
                if "relationship insights" in line.lower():
                    in_insights = True
                    continue
                elif in_insights and line.strip() and not line.startswith("#"):
                    if line.strip().startswith("- ") or line.strip().startswith("1."):
                        insight = (
                            line.strip()[2:]
                            if line.strip().startswith("- ")
                            else line.strip()[2:]
                        )
                        insights.append(insight)
                elif in_insights and len(insights) >= 2:
                    break

            if insights:
                print(f"   💡 Key Insights: {insights[0][:50]}...")


def show_universal_vs_static_comparison():
    """Show the difference between universal templates and generated domain-specific ones"""

    print("\n\n🔄 UNIVERSAL TEMPLATE vs GENERATED TEMPLATE COMPARISON")
    print("=" * 70)

    # Universal template
    universal_entity = Path(
        "/workspace/azure-maintie-rag/infrastructure/prompt_workflows/templates/universal_entity_extraction.jinja2"
    )
    generated_programming = Path(
        "/workspace/azure-maintie-rag/infrastructure/prompt_workflows/generated/programming_tutorial_entity_extraction.jinja2"
    )

    print("\n📋 UNIVERSAL STATIC TEMPLATE (Before Domain Analysis):")
    print("-" * 55)

    if universal_entity.exists():
        universal_content = universal_entity.read_text()
        universal_lines = universal_content.split("\n")[:20]

        for i, line in enumerate(universal_lines, 1):
            if line.strip():
                print(f"   {i:2d}→ {line}")

        print("   ... (template with {{ variables }} for dynamic population)")

    print("\n🎯 GENERATED DOMAIN-SPECIFIC TEMPLATE (After Domain Analysis):")
    print("-" * 60)

    if generated_programming.exists():
        generated_content = generated_programming.read_text()
        generated_lines = generated_content.split("\n")[:20]

        for i, line in enumerate(generated_lines, 1):
            if line.strip():
                print(f"   {i:2d}→ {line}")

        print("   ... (variables populated with domain-specific values)")

    print("\n🔍 KEY DIFFERENCES:")
    print("-" * 20)
    print("   ✅ Universal Template: Contains {{ variables }} for dynamic population")
    print(
        "   ✅ Generated Template: Variables filled with domain-specific analysis results"
    )
    print(
        "   ✅ Domain Intelligence: Automatically discovers entity types, relationships, patterns"
    )
    print(
        "   ✅ Adaptive Configuration: Thresholds, complexity measures adjust to content"
    )
    print(
        "   ✅ Zero Hardcoding: No predetermined domain assumptions - pure content analysis"
    )


def demonstrate_workflow_steps():
    """Show the complete workflow from raw content to generated prompts"""

    print("\n\n🌊 COMPLETE UNIVERSAL RAG WORKFLOW")
    print("=" * 40)

    print(
        """
📄 Step 1: Raw Content Input
   └─ Example: Python programming tutorial, medical research, maintenance procedures

🧠 Step 2: Domain Intelligence Analysis  
   ├─ Content signature discovery (python_programming, medical_research, etc.)
   ├─ Vocabulary complexity calculation (0.1 to 1.0 scale)
   ├─ Concept density analysis (structural complexity)
   └─ Pattern discovery (object_oriented_code, clinical_trial_methodology, etc.)

⚙️ Step 3: Dynamic Configuration Generation
   ├─ Entity types discovered from patterns
   ├─ Relationship patterns based on content structure  
   ├─ Confidence thresholds adapted to vocabulary complexity
   └─ Domain descriptions generated from content characteristics

🎯 Step 4: Template Population & Generation
   ├─ Universal Jinja2 templates loaded
   ├─ Domain-specific variables populated
   ├─ Generated prompts saved with domain signature
   └─ Ready for Knowledge Extraction Agent use

🔄 Step 5: Real-time Adaptation
   ├─ Each new content type triggers fresh analysis
   ├─ No cached domain assumptions
   ├─ Purely content-driven configuration
   └─ Universal approach works for ANY domain
    """
    )

    print("\n✨ UNIVERSAL RAG PHILOSOPHY:")
    print("   🚫 No hardcoded domain categories")
    print("   🚫 No predetermined business logic")
    print("   ✅ Content analysis drives all configuration")
    print("   ✅ Domain characteristics discovered automatically")
    print("   ✅ Templates adapt to ANY content type")
    print("   ✅ Zero domain bias - pure universal approach")


def main():
    """Run the complete prompt adaptation comparison"""
    compare_generated_prompts()
    show_universal_vs_static_comparison()
    demonstrate_workflow_steps()


if __name__ == "__main__":
    main()
