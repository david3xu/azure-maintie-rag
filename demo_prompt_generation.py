#!/usr/bin/env python3
"""
Demo: Universal RAG Dynamic Prompt Generation
============================================

This script demonstrates the actual dynamic prompt generation process:
1. Raw Content Analysis
2. Domain Intelligence Agent Analysis
3. Dynamic Prompt Template Generation
4. Generated Prompt Examples

Shows how Universal RAG discovers domain characteristics and adapts prompts automatically.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# Simple domain analysis simulation (without Azure dependency)
class MockDomainAnalysis:
    """Mock domain analysis results for demonstration"""

    def __init__(self, content: str):
        self.content = content
        self.content_signature = self._analyze_content_signature()
        self.vocabulary_complexity = self._calculate_vocabulary_complexity()
        self.concept_density = self._calculate_concept_density()
        self.discovered_patterns = self._discover_patterns()

    def _analyze_content_signature(self) -> str:
        """Analyze content to discover characteristics without domain assumptions - UNIVERSAL RAG APPROACH"""
        words = self.content.split()
        unique_words = set(word.lower().strip('.,!?;:"()[]') for word in words)

        # Universal content signature based on MEASURED PROPERTIES, not domain labels
        vocab_complexity = min(len(unique_words) / max(len(words), 1), 1.0)

        # Measure concept density
        potential_concepts = [word for word in unique_words if len(word) > 6]
        concept_density = min(len(potential_concepts) / max(len(words), 1) * 10, 1.0)

        # Discover structural patterns (not hardcoded types)
        structural_indicators = 0
        if "```" in self.content or "def " in self.content or "class " in self.content:
            structural_indicators += 1
        if self.content.count("\n- ") > 5 or self.content.count("1. ") > 3:
            structural_indicators += 1
        if self.content.count("\n#") > 2 or self.content.count("##") > 1:
            structural_indicators += 1
        if "|" in self.content and self.content.count("|") > 10:
            structural_indicators += 1

        # Generate signature based on measured properties
        signature_components = [
            f"vc{vocab_complexity:.2f}",  # vocabulary complexity
            f"cd{concept_density:.2f}",  # concept density
            f"sp{structural_indicators}",  # structural patterns count
        ]

        # Universal content signature - no domain assumptions
        return "_".join(signature_components)

    def _calculate_vocabulary_complexity(self) -> float:
        """Calculate vocabulary complexity based on UNIVERSAL MEASUREMENTS, not domain assumptions"""
        words = self.content.split()
        unique_words = set(
            word.lower().strip('.,!?;:"()[]') for word in words if len(word) > 0
        )

        # Universal complexity metrics - no hardcoded domain terms
        long_word_ratio = sum(1 for word in unique_words if len(word) > 8) / max(
            len(unique_words), 1
        )
        capitalized_ratio = sum(
            1 for word in words if word and word[0].isupper()
        ) / max(len(words), 1)
        numeric_ratio = sum(
            1 for word in words if any(char.isdigit() for char in word)
        ) / max(len(words), 1)

        # Combine universal metrics
        complexity = (
            long_word_ratio * 0.5 + capitalized_ratio * 0.3 + numeric_ratio * 0.2
        )
        return min(complexity, 1.0)

    def _calculate_concept_density(self) -> float:
        """Calculate concept density based on UNIVERSAL STRUCTURE ANALYSIS"""
        # Universal structural indicators - no domain assumptions
        bullet_points = self.content.count("\n- ") + self.content.count("\n• ")
        numbered_items = sum(1 for i in range(1, 20) if f"\n{i}. " in self.content)
        colons = self.content.count(":")  # Indicates definitions/explanations
        parentheses = self.content.count("(")  # Indicates clarifications/details

        # Measure sentence structure complexity
        sentences = [s.strip() for s in self.content.split(".") if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(
            len(sentences), 1
        )

        # Universal density calculation
        structural_density = (
            bullet_points + numbered_items + colons / 10 + parentheses / 10
        ) / max(len(sentences), 1)
        length_density = min(avg_sentence_length / 20.0, 1.0)  # Normalize to 0-1

        return min((structural_density + length_density) / 2.0, 1.0)

    def _discover_patterns(self) -> List[str]:
        """Discover content patterns using UNIVERSAL ANALYSIS - no domain assumptions"""
        patterns = []

        # Universal structural patterns based on content organization
        if (
            "```" in self.content
            or self.content.count("def ") > 2
            or self.content.count("class ") > 1
        ):
            patterns.append("code_blocks")

        if self.content.count("\n- ") > 5 or self.content.count("1. ") > 3:
            patterns.append("list_structures")

        if self.content.count("\n#") > 2 or self.content.count("##") > 1:
            patterns.append("hierarchical_headers")

        if "|" in self.content and self.content.count("|") > 10:
            patterns.append("tabular_data")

        # Universal content organization patterns
        if self.content.count(":") > 10:  # Many definitions/explanations
            patterns.append("definition_rich")

        if self.content.count("(") > 20:  # Many clarifications
            patterns.append("detailed_explanations")

        # Sequential patterns based on structure, not content
        numbered_sequences = sum(1 for i in range(1, 10) if f"{i}. " in self.content)
        if numbered_sequences > 5:
            patterns.append("sequential_steps")

        # Reference patterns
        if (
            self.content.count("°") > 0
            or self.content.count("PSI") > 0
            or self.content.count("mg/") > 0
        ):
            patterns.append("measurement_data")

        return patterns


class DynamicPromptDemo:
    """Demonstrates dynamic prompt generation without Azure dependencies"""

    def __init__(self):
        self.template_dir = (
            Path(__file__).parent / "infrastructure/prompt_workflows/templates"
        )
        self.generated_dir = (
            Path(__file__).parent / "infrastructure/prompt_workflows/generated"
        )
        self.generated_dir.mkdir(exist_ok=True)

        self.jinja_env = Environment(loader=FileSystemLoader(str(self.template_dir)))

    async def run_complete_demo(self):
        """Run complete demonstration of dynamic prompt generation"""
        print("🌍 Universal RAG: Dynamic Prompt Generation Demo")
        print("=" * 60)
        print()

        # Load sample content
        data_dir = Path(__file__).parent / "data/raw"
        sample_files = [
            "programming_tutorial.txt",
            "medical_research.txt",
            "maintenance_procedures.txt",
        ]

        for sample_file in sample_files:
            file_path = data_dir / sample_file
            if file_path.exists():
                print(f"📄 Processing: {sample_file}")
                content = file_path.read_text()
                await self.process_content_example(
                    content, sample_file.replace(".txt", "")
                )
                print()

        # Show generated files
        await self.show_generated_files()

    async def process_content_example(self, content: str, content_name: str):
        """Process a single content example through the complete workflow"""

        print(f"   Raw Content Length: {len(content)} characters")
        print(f"   Content Preview: {content[:100]}...")
        print()

        # Step 1: Domain Intelligence Analysis
        print("   🧠 Step 1: Domain Intelligence Analysis")
        domain_analysis = MockDomainAnalysis(content)

        print(f"      ✅ Discovered Domain: {domain_analysis.content_signature}")
        print(
            f"      ✅ Vocabulary Complexity: {domain_analysis.vocabulary_complexity:.3f}"
        )
        print(f"      ✅ Concept Density: {domain_analysis.concept_density:.3f}")
        print(
            f"      ✅ Discovered Patterns: {', '.join(domain_analysis.discovered_patterns)}"
        )
        print()

        # Step 2: Dynamic Configuration Generation
        print("   ⚙️ Step 2: Dynamic Configuration Generation")
        config = self._generate_prompt_config(domain_analysis)

        print(
            f"      ✅ Entity Types: {', '.join(config['discovered_entity_types'][:3])}..."
        )
        print(
            f"      ✅ Relationship Types: {', '.join(config['discovered_relationship_types'][:3])}..."
        )
        print(
            f"      ✅ Confidence Thresholds: Entity={config['entity_confidence_threshold']:.2f}, Relation={config['relationship_confidence_threshold']:.2f}"
        )
        print()

        # Step 3: Dynamic Prompt Generation
        print("   🎯 Step 3: Dynamic Prompt Template Generation")
        generated_prompts = await self._generate_domain_prompts(config, content_name)

        for prompt_type, file_path in generated_prompts.items():
            print(f"      ✅ Generated {prompt_type}: {Path(file_path).name}")
        print()

        return generated_prompts

    def _generate_prompt_config(
        self, domain_analysis: MockDomainAnalysis
    ) -> Dict[str, Any]:
        """Generate prompt configuration from domain analysis"""

        # Universal entity type discovery based on STRUCTURAL PATTERNS, not domain assumptions
        entity_types = ["concept", "term", "item"]  # Always start with universal types

        # Add types based on discovered structural patterns
        if "code_blocks" in domain_analysis.discovered_patterns:
            entity_types.extend(["component", "element", "structure"])
        if "list_structures" in domain_analysis.discovered_patterns:
            entity_types.extend(["list_item", "category", "group"])
        if "hierarchical_headers" in domain_analysis.discovered_patterns:
            entity_types.extend(["section", "subsection", "topic"])
        if "definition_rich" in domain_analysis.discovered_patterns:
            entity_types.extend(["definition", "explanation", "description"])
        if "sequential_steps" in domain_analysis.discovered_patterns:
            entity_types.extend(["step", "stage", "phase"])
        if "measurement_data" in domain_analysis.discovered_patterns:
            entity_types.extend(["measurement", "value", "parameter"])

        # Universal fallback
        if len(entity_types) == 3:  # Only base types
            entity_types.extend(["object", "attribute", "property"])

        # Universal relationship pattern discovery based on STRUCTURAL CHARACTERISTICS
        relationship_patterns = []

        # High complexity content patterns
        if domain_analysis.vocabulary_complexity > 0.6:
            relationship_patterns.append(
                {
                    "category": "Complex",
                    "relationship_types": [
                        {
                            "name": "relates_to",
                            "description": "Complex relationship",
                            "example": "concept relates to framework",
                        },
                        {
                            "name": "depends_on",
                            "description": "Dependency relationship",
                            "example": "element depends on component",
                        },
                        {
                            "name": "configures",
                            "description": "Configuration relationship",
                            "example": "parameter configures behavior",
                        },
                    ],
                }
            )

        # Sequential structure patterns
        if "sequential_steps" in domain_analysis.discovered_patterns:
            relationship_patterns.append(
                {
                    "category": "Sequential",
                    "relationship_types": [
                        {
                            "name": "precedes",
                            "description": "Sequential relationship",
                            "example": "step precedes next_step",
                        },
                        {
                            "name": "enables",
                            "description": "Enabling relationship",
                            "example": "action enables outcome",
                        },
                        {
                            "name": "requires",
                            "description": "Requirement relationship",
                            "example": "process requires resource",
                        },
                    ],
                }
            )

        # Hierarchical structure patterns
        if "hierarchical_headers" in domain_analysis.discovered_patterns:
            relationship_patterns.append(
                {
                    "category": "Hierarchical",
                    "relationship_types": [
                        {
                            "name": "contains",
                            "description": "Containment relationship",
                            "example": "section contains subsection",
                        },
                        {
                            "name": "belongs_to",
                            "description": "Membership relationship",
                            "example": "item belongs to category",
                        },
                        {
                            "name": "composed_of",
                            "description": "Composition relationship",
                            "example": "whole composed of parts",
                        },
                    ],
                }
            )

        # Universal relationship patterns
        relationship_patterns.append(
            {
                "category": "Conceptual",
                "relationship_types": [
                    {
                        "name": "relates_to",
                        "description": "General relationship",
                        "example": "concept relates to theme",
                    },
                    {
                        "name": "contains",
                        "description": "Containment relationship",
                        "example": "category contains items",
                    },
                    {
                        "name": "associated_with",
                        "description": "Association relationship",
                        "example": "term associated with concept",
                    },
                ],
            }
        )

        # Extract all relationship type names
        relationship_types = []
        for pattern in relationship_patterns:
            for rel_type in pattern["relationship_types"]:
                relationship_types.append(rel_type["name"])

        return {
            "domain_signature": domain_analysis.content_signature,
            "content_confidence": min(
                domain_analysis.vocabulary_complexity + domain_analysis.concept_density,
                1.0,
            ),
            "discovered_domain_description": self._generate_domain_description(
                domain_analysis
            ),
            "discovered_content_patterns": [
                {
                    "category": pattern.replace("_", " ").title(),
                    "description": f"Discovered {pattern.replace('_', ' ')} pattern in content",
                    "examples": [f"{pattern}_example_1", f"{pattern}_example_2"],
                }
                for pattern in domain_analysis.discovered_patterns[:3]
            ],
            "discovered_entity_types": entity_types,
            "discovered_relationship_patterns": relationship_patterns,
            "discovered_relationship_types": relationship_types,
            "entity_confidence_threshold": max(
                0.6, domain_analysis.vocabulary_complexity * 0.8
            ),
            "relationship_confidence_threshold": max(
                0.5, domain_analysis.vocabulary_complexity * 0.7
            ),
            "key_domain_insights": [
                f"Content signature: {domain_analysis.content_signature}",
                f"Vocabulary complexity: {domain_analysis.vocabulary_complexity:.2f}",
                f"Concept density: {domain_analysis.concept_density:.2f}",
                f"Discovered {len(domain_analysis.discovered_patterns)} patterns",
                "Universal domain-agnostic extraction approach",
            ],
            "relationship_insights": [
                "How entities relate based on content structure",
                "Universal relationship patterns discovered from analysis",
                "Content-agnostic interaction patterns",
            ],
            "vocabulary_richness": domain_analysis.vocabulary_complexity,
            "technical_density": domain_analysis.concept_density,
            "analysis_processing_time": 0.1,
            "example_entity": "discovered_concept",
            "adaptive_entity_type": entity_types[0] if entity_types else "concept",
        }

    def _generate_domain_description(self, domain_analysis: MockDomainAnalysis) -> str:
        """Generate universal content description based on MEASURED CHARACTERISTICS"""
        if domain_analysis.vocabulary_complexity > 0.7:
            return f"highly complex content with sophisticated vocabulary"
        elif domain_analysis.vocabulary_complexity > 0.5:
            return f"moderately complex content with specialized vocabulary"
        else:
            return f"straightforward content with standard vocabulary"

    async def _generate_domain_prompts(
        self, config: Dict[str, Any], content_name: str
    ) -> Dict[str, str]:
        """Generate domain-specific prompts using configuration"""
        generated_prompts = {}

        # Generate entity extraction prompt
        entity_template = self.jinja_env.get_template(
            "universal_entity_extraction.jinja2"
        )
        entity_prompt = entity_template.render(**config)

        entity_path = self.generated_dir / f"{content_name}_entity_extraction.jinja2"
        entity_path.write_text(entity_prompt)
        generated_prompts["entity_extraction"] = str(entity_path)

        # Generate relationship extraction prompt
        relation_template = self.jinja_env.get_template(
            "universal_relation_extraction.jinja2"
        )
        relation_prompt = relation_template.render(**config)

        relation_path = (
            self.generated_dir / f"{content_name}_relation_extraction.jinja2"
        )
        relation_path.write_text(relation_prompt)
        generated_prompts["relation_extraction"] = str(relation_path)

        return generated_prompts

    async def show_generated_files(self):
        """Show the actual generated prompt files"""
        print("📁 Generated Domain-Specific Prompt Templates")
        print("=" * 50)

        generated_files = list(self.generated_dir.glob("*.jinja2"))

        if not generated_files:
            print("   No generated files found.")
            return

        for file_path in sorted(generated_files):
            print(f"\n📄 {file_path.name}")
            print("-" * len(file_path.name))

            # Show first 15 lines of each generated file
            content = file_path.read_text()
            lines = content.split("\n")[:15]
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}→ {line}")

            total_lines = len(content.split("\n"))
            if total_lines > 15:
                print(f"   ... (showing first 15 lines of {total_lines} total)")

            print()


async def main():
    """Run the complete dynamic prompt generation demo"""
    demo = DynamicPromptDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
