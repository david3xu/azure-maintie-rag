#!/usr/bin/env python3
"""
Universal Prompt Workflow Generator - Zero Domain Bias
====================================================

Automatically generates domain-specific prompt workflows using Universal Domain Intelligence.
No hardcoded domain assumptions - adapts to ANY content type.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Removed circular import - domain analysis should be injected as dependency
# from agents.domain_intelligence.agent import run_universal_domain_analysis, UniversalDomainDeps


class UniversalPromptGenerator:
    """
    Generates domain-specific prompt workflows automatically using Universal Domain Intelligence.

    Zero hardcoded assumptions - all configuration comes from actual content analysis.
    """

    def __init__(self, template_directory: str = None):
        """Initialize universal prompt generator"""
        if template_directory is None:
            template_directory = str(Path(__file__).parent)

        self.template_env = Environment(loader=FileSystemLoader(template_directory))
        self.templates = {
            "entity_extraction": "universal_entity_extraction.jinja2",
            "relation_extraction": "universal_relation_extraction.jinja2",
        }

    async def generate_domain_prompts(
        self,
        data_directory: str,
        output_directory: str = None,
        max_files_to_analyze: int = 50,
    ) -> Dict[str, str]:
        """
        Generate domain-specific prompt workflows automatically.

        Steps:
        1. Run universal domain analysis on content
        2. Extract domain characteristics and patterns
        3. Generate adaptive prompt templates
        4. Save configured prompts for the discovered domain

        Returns dictionary of generated prompt file paths.
        """
        print(f"🌍 Generating universal prompts for content in: {data_directory}")

        # Step 1: Universal Domain Analysis
        print("📊 Running universal domain analysis...")
        deps = UniversalDomainDeps(
            data_directory=data_directory,
            max_files_to_analyze=max_files_to_analyze,
            min_content_length=100,
            enable_multilingual=True,
        )

        domain_analysis = await run_universal_domain_analysis(deps)

        print(f"✅ Domain discovered: {domain_analysis.domain_signature}")
        print(f"   Content confidence: {domain_analysis.content_type_confidence:.2f}")
        print(
            f"   Vocabulary richness: {domain_analysis.characteristics.vocabulary_richness:.3f}"
        )
        print(
            f"   Technical density: {domain_analysis.characteristics.technical_vocabulary_ratio:.3f}"
        )

        # Step 2: Extract Domain Intelligence for Prompt Configuration
        prompt_config = self._extract_prompt_configuration(domain_analysis)

        # Step 3: Generate Adaptive Prompts
        generated_prompts = {}

        if output_directory is None:
            output_directory = str(Path(__file__).parent / "generated")

        Path(output_directory).mkdir(exist_ok=True)

        # Generate Entity Extraction Prompt
        entity_prompt = self._generate_entity_extraction_prompt(prompt_config)
        entity_path = (
            Path(output_directory)
            / f"{domain_analysis.domain_signature}_entity_extraction.jinja2"
        )
        entity_path.write_text(entity_prompt)
        generated_prompts["entity_extraction"] = str(entity_path)

        # Generate Relation Extraction Prompt
        relation_prompt = self._generate_relation_extraction_prompt(prompt_config)
        relation_path = (
            Path(output_directory)
            / f"{domain_analysis.domain_signature}_relation_extraction.jinja2"
        )
        relation_path.write_text(relation_prompt)
        generated_prompts["relation_extraction"] = str(relation_path)

        print(f"✅ Generated domain-specific prompts:")
        for prompt_type, path in generated_prompts.items():
            print(f"   {prompt_type}: {path}")

        return generated_prompts

    def _extract_prompt_configuration(self, domain_analysis) -> Dict[str, Any]:
        """Extract configuration for prompt generation from domain analysis"""

        # Discover entity types from content patterns
        discovered_entity_types = []
        discovered_content_patterns = []

        if domain_analysis.characteristics.content_patterns:
            for pattern in domain_analysis.characteristics.content_patterns[
                :5
            ]:  # Top 5 patterns
                discovered_content_patterns.append(
                    {
                        "category": pattern.replace("_", " ").title(),
                        "description": f"Key {pattern.replace('_', ' ')} found in domain",
                        "examples": domain_analysis.characteristics.most_frequent_terms[
                            :3
                        ],
                    }
                )
                discovered_entity_types.append(pattern.replace("_", " ").lower())

        # Generate relationship patterns from domain characteristics
        relationship_patterns = self._discover_relationship_patterns(domain_analysis)

        # Create adaptive configuration
        return {
            "domain_signature": domain_analysis.domain_signature,
            "content_confidence": domain_analysis.content_type_confidence,
            "discovered_domain_description": self._generate_domain_description(
                domain_analysis
            ),
            "discovered_content_patterns": discovered_content_patterns,
            "discovered_entity_types": discovered_entity_types
            or ["concept", "entity", "term"],
            "discovered_relationship_patterns": relationship_patterns,
            "discovered_relationship_types": self._extract_relationship_types(
                relationship_patterns
            ),
            "entity_confidence_threshold": domain_analysis.processing_config.entity_confidence_threshold,
            "relationship_confidence_threshold": max(
                0.7, domain_analysis.processing_config.entity_confidence_threshold - 0.1
            ),
            "key_domain_insights": domain_analysis.key_insights[:5],
            "relationship_insights": self._generate_relationship_insights(
                domain_analysis
            ),
            "vocabulary_richness": domain_analysis.characteristics.vocabulary_richness,
            "technical_density": domain_analysis.characteristics.technical_vocabulary_ratio,
            "analysis_processing_time": domain_analysis.processing_time,
            "example_entity": (
                domain_analysis.characteristics.most_frequent_terms[0]
                if domain_analysis.characteristics.most_frequent_terms
                else "discovered_entity"
            ),
            "adaptive_entity_type": (
                discovered_entity_types[0] if discovered_entity_types else "concept"
            ),
        }

    def _generate_domain_description(self, domain_analysis) -> str:
        """Generate natural language description of discovered content characteristics"""
        signature = domain_analysis.domain_signature.replace("_", " ")

        if domain_analysis.characteristics.vocabulary_complexity > 0.3:
            return f"specialized terminology {signature} content"
        elif domain_analysis.characteristics.vocabulary_richness > 0.4:
            return f"specialized {signature} content"
        else:
            return f"{signature} content"

    def _discover_relationship_patterns(self, domain_analysis) -> List[Dict[str, Any]]:
        """Discover relationship patterns from domain characteristics"""
        patterns = []

        # Base on content complexity and terminology density
        if domain_analysis.characteristics.vocabulary_complexity > 0.3:
            patterns.append(
                {
                    "category": "Structured",
                    "relationship_types": [
                        {
                            "name": "implements",
                            "description": "One entity implements another",
                            "example": "method implements interface",
                        },
                        {
                            "name": "depends_on",
                            "description": "Dependency relationship",
                            "example": "component depends on library",
                        },
                        {
                            "name": "configures",
                            "description": "Configuration relationship",
                            "example": "parameter configures behavior",
                        },
                    ],
                }
            )

        if (
            "process" in domain_analysis.domain_signature
            or domain_analysis.characteristics.sentence_complexity > 15
        ):
            patterns.append(
                {
                    "category": "Process",
                    "relationship_types": [
                        {
                            "name": "leads_to",
                            "description": "Sequential relationship",
                            "example": "step leads to outcome",
                        },
                        {
                            "name": "enables",
                            "description": "Enabling relationship",
                            "example": "tool enables process",
                        },
                        {
                            "name": "requires",
                            "description": "Requirement relationship",
                            "example": "process requires resource",
                        },
                    ],
                }
            )

        # Default conceptual patterns for any domain
        patterns.append(
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

        return patterns

    def _extract_relationship_types(
        self, relationship_patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract relationship type names from patterns"""
        types = []
        for pattern in relationship_patterns:
            for rel_type in pattern["relationship_types"]:
                types.append(rel_type["name"])
        return types

    def _generate_relationship_insights(self, domain_analysis) -> List[str]:
        """Generate relationship insights from domain analysis"""
        insights = [
            f"How entities are structured in {domain_analysis.domain_signature}",
            f"What relationships exist between key concepts",
            f"How information flows within the domain",
        ]

        if domain_analysis.characteristics.vocabulary_complexity > 0.3:
            insights.append("Structured dependencies and implementation relationships")

        if domain_analysis.characteristics.sentence_complexity > 15:
            insights.append("Process flows and sequential relationships")

        return insights

    def _generate_entity_extraction_prompt(self, config: Dict[str, Any]) -> str:
        """Generate entity extraction prompt using configuration"""
        template = self.template_env.get_template(self.templates["entity_extraction"])
        return template.render(**config)

    def _generate_relation_extraction_prompt(self, config: Dict[str, Any]) -> str:
        """Generate relation extraction prompt using configuration"""
        template = self.template_env.get_template(self.templates["relation_extraction"])
        return template.render(**config)


async def main():
    """Demo: Generate domain-specific prompts from actual content"""
    generator = UniversalPromptGenerator()

    # Generate prompts for your actual data
    data_directory = "/workspace/azure-maintie-rag/data/raw"

    print("🌍 Universal Prompt Workflow Generator")
    print("=====================================")
    print("Generating domain-specific prompts without hardcoded assumptions...")

    generated_prompts = await generator.generate_domain_prompts(
        data_directory=data_directory, max_files_to_analyze=20
    )

    print("\n✅ Universal prompt generation completed!")
    print("These prompts will adapt to ANY domain you provide.")

    return generated_prompts


if __name__ == "__main__":
    asyncio.run(main())
