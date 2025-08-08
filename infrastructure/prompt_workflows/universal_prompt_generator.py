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
from typing import Any, Dict, List

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

    def __init__(self, template_directory: str = None, domain_analyzer=None):
        """Initialize universal prompt generator with dependency injection"""
        if template_directory is None:
            template_directory = str(Path(__file__).parent / "templates")

        self.template_env = Environment(loader=FileSystemLoader(template_directory))
        self.templates = {
            "entity_extraction": "universal_entity_extraction.jinja2",
            "relation_extraction": "universal_relation_extraction.jinja2",
        }
        self.domain_analyzer = domain_analyzer  # Injected dependency

    async def inject_domain_analyzer(self, domain_analyzer):
        """Inject domain analyzer after initialization"""
        self.domain_analyzer = domain_analyzer
        return self

    @classmethod
    async def create_with_domain_intelligence(cls, template_directory: str = None):
        """Create prompt generator with proper domain intelligence injection"""
        from agents.domain_intelligence.agent import run_domain_analysis

        instance = cls(template_directory=template_directory)
        instance.domain_analyzer = run_domain_analysis
        return instance

    async def generate_domain_prompts(
        self,
        data_directory: str,
        output_directory: str = None,
        max_files_to_analyze: int = 50,
    ) -> Dict[str, str]:
        """
        Generate domain-specific prompt workflows automatically.

        Steps:
        1. Run universal domain analysis on content (via injected analyzer)
        2. Extract domain characteristics and patterns
        3. Generate adaptive prompt templates
        4. Save configured prompts for the discovered domain

        Returns dictionary of generated prompt file paths.
        """
        if self.domain_analyzer is None:
            print(
                "⚠️  Domain analyzer not injected. Using universal fallback templates..."
            )
            return await self._generate_fallback_prompts(
                output_directory or str(Path(__file__).parent / "generated")
            )

        print(f"🌍 Generating universal prompts for content in: {data_directory}")

        # Step 1: Universal Domain Analysis (via dependency injection)
        print("📊 Running universal domain analysis...")

        # Analyze sample content from directory for domain characteristics
        content_sample = await self._extract_sample_content(
            data_directory, max_files_to_analyze
        )

        # Use injected domain analyzer
        domain_analysis = await self.domain_analyzer(content_sample, detailed=True)

        print(f"✅ Domain discovered: {domain_analysis.content_signature}")
        print(f"   Vocabulary complexity: {domain_analysis.vocabulary_complexity:.2f}")
        print(f"   Concept density: {domain_analysis.concept_density:.2f}")
        print(f"   Discovered patterns: {len(domain_analysis.discovered_patterns)}")

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
            / f"{domain_analysis.content_signature}_entity_extraction.jinja2"
        )
        entity_path.write_text(entity_prompt)
        generated_prompts["entity_extraction"] = str(entity_path)

        # Generate Relation Extraction Prompt
        relation_prompt = self._generate_relation_extraction_prompt(prompt_config)
        relation_path = (
            Path(output_directory)
            / f"{domain_analysis.content_signature}_relation_extraction.jinja2"
        )
        relation_path.write_text(relation_prompt)
        generated_prompts["relation_extraction"] = str(relation_path)

        print(f"✅ Generated domain-specific prompts:")
        for prompt_type, path in generated_prompts.items():
            print(f"   {prompt_type}: {path}")

        return generated_prompts

    async def _extract_sample_content(self, data_directory: str, max_files: int) -> str:
        """Extract sample content from directory for domain analysis"""
        data_path = Path(data_directory)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_directory}")

        content_parts = []
        file_count = 0

        # Collect sample content from files
        for file_path in data_path.rglob("*.txt"):
            if file_count >= max_files:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content) > 100:  # Minimum content length
                    content_parts.append(content[:1000])  # Sample first 1000 chars
                    file_count += 1
            except Exception:
                continue

        # Also check for other common formats
        for file_path in data_path.rglob("*.md"):
            if file_count >= max_files:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(content) > 100:
                    content_parts.append(content[:1000])
                    file_count += 1
            except Exception:
                continue

        if not content_parts:
            return "Sample content for domain analysis"

        return "\n\n".join(content_parts[:max_files])

    def _extract_prompt_configuration(self, domain_analysis) -> Dict[str, Any]:
        """Extract configuration for prompt generation from domain analysis"""

        # Work with actual UniversalDomainAnalysis model
        discovered_patterns = getattr(domain_analysis, "discovered_patterns", [])
        content_signature = getattr(domain_analysis, "content_signature", "unknown")
        vocabulary_complexity = getattr(domain_analysis, "vocabulary_complexity", 0.5)
        concept_density = getattr(domain_analysis, "concept_density", 0.5)

        # Discover entity types from discovered patterns
        discovered_entity_types = []
        discovered_content_patterns = []

        for i, pattern in enumerate(discovered_patterns[:5]):  # Top 5 patterns
            pattern_name = pattern.replace("_", " ").title()
            discovered_content_patterns.append(
                {
                    "category": pattern_name,
                    "description": f"Key {pattern.replace('_', ' ')} pattern found in content",
                    "examples": [f"example_{i}_1", f"example_{i}_2", f"example_{i}_3"],
                }
            )
            discovered_entity_types.append(pattern.replace("_", " ").lower())

        # Generate relationship patterns from domain characteristics
        relationship_patterns = self._discover_relationship_patterns_simple(
            vocabulary_complexity, discovered_patterns
        )

        # Create adaptive configuration based on actual domain analysis results
        return {
            "domain_signature": content_signature,
            "content_confidence": min(vocabulary_complexity + concept_density, 1.0),
            "discovered_domain_description": self._generate_domain_description_simple(
                content_signature, vocabulary_complexity
            ),
            "discovered_content_patterns": discovered_content_patterns,
            "discovered_entity_types": discovered_entity_types
            or ["concept", "entity", "term"],
            "discovered_relationship_patterns": relationship_patterns,
            "discovered_relationship_types": self._extract_relationship_types(
                relationship_patterns
            ),
            "entity_confidence_threshold": max(0.6, vocabulary_complexity * 0.8),
            "relationship_confidence_threshold": max(0.5, vocabulary_complexity * 0.7),
            "key_domain_insights": [
                f"Content signature: {content_signature}",
                f"Vocabulary complexity: {vocabulary_complexity:.2f}",
                f"Concept density: {concept_density:.2f}",
                f"Discovered {len(discovered_patterns)} patterns",
                "Universal domain-agnostic extraction approach",
            ],
            "relationship_insights": self._generate_relationship_insights_simple(
                content_signature, discovered_patterns
            ),
            "vocabulary_richness": vocabulary_complexity,
            "technical_density": concept_density,
            "analysis_processing_time": 0.0,
            "example_entity": "discovered_concept",
            "adaptive_entity_type": (
                discovered_entity_types[0] if discovered_entity_types else "concept"
            ),
        }

    def _generate_domain_description_simple(
        self, content_signature: str, vocabulary_complexity: float
    ) -> str:
        """Generate natural language description of discovered content characteristics"""
        signature = content_signature.replace("_", " ")

        if vocabulary_complexity > 0.7:
            return f"complex specialized {signature} content"
        elif vocabulary_complexity > 0.5:
            return f"specialized {signature} content"
        else:
            return f"{signature} content"

    def _discover_relationship_patterns_simple(
        self, vocabulary_complexity: float, discovered_patterns: List[str]
    ) -> List[Dict[str, Any]]:
        """Discover relationship patterns from domain characteristics"""
        patterns = []

        # Base on content complexity and discovered patterns
        if vocabulary_complexity > 0.6 or "code_blocks" in discovered_patterns:
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

        if "hierarchical_headers" in discovered_patterns or vocabulary_complexity > 0.5:
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

    def _generate_relationship_insights_simple(
        self, content_signature: str, discovered_patterns: List[str]
    ) -> List[str]:
        """Generate relationship insights from domain analysis"""
        insights = [
            f"How entities are structured in {content_signature}",
            f"What relationships exist between key concepts",
            f"How information flows within the content",
        ]

        if "code_blocks" in discovered_patterns:
            insights.append("Structured dependencies and implementation relationships")

        if "hierarchical_headers" in discovered_patterns:
            insights.append("Process flows and sequential relationships")

        return insights

    async def _generate_fallback_prompts(self, output_directory: str) -> Dict[str, str]:
        """Generate universal fallback prompts when domain analysis is not available."""
        print("🔧 Generating universal fallback prompts (no domain analysis)...")

        Path(output_directory).mkdir(exist_ok=True)

        # Universal fallback configuration
        fallback_config = {
            "domain_signature": "universal_fallback",
            "content_confidence": 0.7,
            "discovered_domain_description": "universal content without domain-specific analysis",
            "discovered_content_patterns": [
                {
                    "category": "General",
                    "description": "Universal content pattern",
                    "examples": ["concept", "entity", "term"],
                }
            ],
            "discovered_entity_types": ["concept", "entity", "term", "proper_noun"],
            "discovered_relationship_patterns": [
                {
                    "category": "Universal",
                    "relationship_types": [
                        {
                            "name": "relates_to",
                            "description": "General relationship",
                            "example": "A relates to B",
                        },
                        {
                            "name": "contains",
                            "description": "Containment relationship",
                            "example": "A contains B",
                        },
                        {
                            "name": "associated_with",
                            "description": "Association relationship",
                            "example": "A associated with B",
                        },
                    ],
                }
            ],
            "discovered_relationship_types": [
                "relates_to",
                "contains",
                "associated_with",
            ],
            "entity_confidence_threshold": 0.7,
            "relationship_confidence_threshold": 0.6,
            "key_domain_insights": [
                "Universal content analysis without domain assumptions",
                "Domain-agnostic entity and relationship extraction",
                "Fallback approach for any content type",
                "Adaptive to discovered content characteristics",
                "No predetermined domain categories",
            ],
            "relationship_insights": [
                "How entities relate in general content",
                "Universal relationship patterns",
                "Content-agnostic associations",
            ],
            "vocabulary_richness": 0.5,
            "technical_density": 0.5,
            "analysis_processing_time": 0.0,
            "example_entity": "example_concept",
            "adaptive_entity_type": "concept",
        }

        generated_prompts = {}

        # Generate fallback entity extraction prompt
        try:
            entity_prompt = self._generate_entity_extraction_prompt(fallback_config)
            entity_path = (
                Path(output_directory) / "universal_fallback_entity_extraction.jinja2"
            )
            entity_path.write_text(entity_prompt)
            generated_prompts["entity_extraction"] = str(entity_path)
        except Exception as e:
            print(f"⚠️  Failed to generate entity extraction prompt: {e}")
            # Use the universal template directly
            generated_prompts["entity_extraction"] = str(
                Path(self.template_env.loader.searchpath[0])
                / "universal_entity_extraction.jinja2"
            )

        # Generate fallback relation extraction prompt
        try:
            relation_prompt = self._generate_relation_extraction_prompt(fallback_config)
            relation_path = (
                Path(output_directory) / "universal_fallback_relation_extraction.jinja2"
            )
            relation_path.write_text(relation_prompt)
            generated_prompts["relation_extraction"] = str(relation_path)
        except Exception as e:
            print(f"⚠️  Failed to generate relation extraction prompt: {e}")
            # Use the universal template directly
            generated_prompts["relation_extraction"] = str(
                Path(self.template_env.loader.searchpath[0])
                / "universal_relation_extraction.jinja2"
            )

        print(f"✅ Generated fallback prompts:")
        for prompt_type, path in generated_prompts.items():
            print(f"   {prompt_type}: {path}")

        return generated_prompts

    def _generate_entity_extraction_prompt(self, config: Dict[str, Any]) -> str:
        """Generate entity extraction prompt using configuration"""
        template = self.template_env.get_template(self.templates["entity_extraction"])
        return template.render(**config)

    def _generate_relation_extraction_prompt(self, config: Dict[str, Any]) -> str:
        """Generate relation extraction prompt using configuration"""
        template = self.template_env.get_template(self.templates["relation_extraction"])
        return template.render(**config)

    async def cleanup_generated_templates(
        self, generated_directory: str = None, max_age_hours: int = 24
    ):
        """Clean up old generated templates based on age"""
        import os
        import time

        if generated_directory is None:
            generated_directory = str(Path(__file__).parent / "generated")

        generated_path = Path(generated_directory)
        if not generated_path.exists():
            return

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        cleaned_count = 0
        for template_file in generated_path.glob("*.jinja2"):
            file_age = current_time - os.path.getmtime(template_file)
            if file_age > max_age_seconds:
                template_file.unlink()
                cleaned_count += 1

        if cleaned_count > 0:
            print(
                f"🧹 Cleaned up {cleaned_count} old generated templates (older than {max_age_hours}h)"
            )

    def rotate_templates(
        self,
        domain_signature: str,
        generated_directory: str = None,
        max_versions: int = 3,
    ):
        """Implement template versioning and rotation for a domain"""
        if generated_directory is None:
            generated_directory = str(Path(__file__).parent / "generated")

        generated_path = Path(generated_directory)
        if not generated_path.exists():
            return

        # Find existing versions for this domain
        pattern = f"{domain_signature}_*_v*.jinja2"
        existing_versions = sorted(generated_path.glob(pattern))

        # Remove oldest versions if we exceed max_versions
        if len(existing_versions) >= max_versions:
            for old_version in existing_versions[: -max_versions + 1]:
                old_version.unlink()
                print(f"🔄 Rotated old template version: {old_version.name}")

    def list_generated_templates(
        self, generated_directory: str = None
    ) -> Dict[str, List[str]]:
        """List all generated templates organized by domain"""
        if generated_directory is None:
            generated_directory = str(Path(__file__).parent / "generated")

        generated_path = Path(generated_directory)
        if not generated_path.exists():
            return {}

        templates_by_domain = {}
        for template_file in generated_path.glob("*.jinja2"):
            domain = template_file.name.split("_")[0]
            if domain not in templates_by_domain:
                templates_by_domain[domain] = []
            templates_by_domain[domain].append(str(template_file))

        return templates_by_domain


async def main():
    """Demo: Generate domain-specific prompts from actual content"""
    # Create generator with proper domain intelligence injection
    generator = await UniversalPromptGenerator.create_with_domain_intelligence()

    # Generate prompts for your actual data
    data_directory = "/workspace/azure-maintie-rag/data/raw"

    print("🌍 Universal Prompt Workflow Generator")
    print("=====================================")
    print("Generating domain-specific prompts without hardcoded assumptions...")

    try:
        generated_prompts = await generator.generate_domain_prompts(
            data_directory=data_directory, max_files_to_analyze=20
        )

        print("\n✅ Universal prompt generation completed!")
        print("These prompts will adapt to ANY domain you provide.")
        return generated_prompts

    except Exception as e:
        print(f"❌ Error generating prompts: {e}")
        return {}
