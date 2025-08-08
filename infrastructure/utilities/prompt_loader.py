"""
Prompt Template Loader
Loads and renders Jinja2 prompt templates from prompt_flows directory
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

logger = logging.getLogger(__name__)


class PromptTemplateLoader:
    """Load and render Jinja2 prompt templates"""

    def __init__(self):
        # Get prompt flows directory
        self.base_dir = Path(__file__).parent.parent.parent
        self.prompt_flows_dir = self.base_dir / "prompt_flows"
        self.knowledge_extraction_dir = (
            self.prompt_flows_dir / "universal_knowledge_extraction"
        )

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.knowledge_extraction_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def load_template(self, template_name: str) -> Optional[Template]:
        """Load a Jinja2 template by name"""
        try:
            return self.env.get_template(template_name)
        except TemplateNotFound as e:
            logger.error(
                f"Template not found: {template_name} in {self.knowledge_extraction_dir}"
            )
            return None
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            return None

    def render_knowledge_extraction_prompt(
        self,
        text_content: str,
        domain_name: str = "universal",  # Domain-agnostic default
        extraction_focus: str = None,
    ) -> str:
        """
        Render the direct knowledge extraction prompt template

        Args:
            text_content: The text to extract knowledge from
            domain_name: Domain for extraction (e.g., 'maintenance')
            extraction_focus: Comma-separated focus areas

        Returns:
            Rendered prompt string ready for Azure OpenAI
        """
        try:
            # Get extraction focus from domain patterns if not provided
            if not extraction_focus:
                # Use simple domain-based focus to avoid circular imports
                extraction_focus = (
                    f"entities, relationships, {domain_name}-specific concepts"
                )

            # Load template
            template = self.load_template("direct_knowledge_extraction.jinja2")
            if not template:
                # Fallback to hardcoded prompt if template not found
                logger.warning("Using fallback hardcoded prompt")
                return self._fallback_hardcoded_prompt(
                    text_content, domain_name, extraction_focus
                )

            # Render with variables
            rendered = template.render(
                text_content=text_content,
                domain_name=domain_name,
                extraction_focus=extraction_focus,
            )

            return rendered.strip()

        except Exception as e:
            logger.error(f"Error rendering knowledge extraction prompt: {e}")
            # Fallback to hardcoded prompt
            return self._fallback_hardcoded_prompt(
                text_content, domain_name, extraction_focus
            )

    def _fallback_hardcoded_prompt(
        self, text_content: str, domain_name: str, extraction_focus: str
    ) -> str:
        """Fallback to original hardcoded prompt if template loading fails"""
        return f"""You are a knowledge extraction system. Extract entities and relationships from this {domain_name} text.

Text: {text_content}

IMPORTANT: You MUST respond with valid JSON only. No additional text or explanations.

Required JSON format:
{{
  "entities": [
    {{"text": "entity_name", "type": "entity_type", "context": "surrounding_context"}}
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "relation": "relationship_type", "context": "context"}}
  ]
}}

Focus on: {extraction_focus}.
If no clear entities exist, return empty arrays but maintain JSON format."""

    def list_available_templates(self) -> list:
        """List all available template files"""
        try:
            template_files = []
            if self.knowledge_extraction_dir.exists():
                for file_path in self.knowledge_extraction_dir.glob("*.jinja2"):
                    template_files.append(file_path.name)
            return sorted(template_files)
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []


# Global instance for easy access
prompt_loader = PromptTemplateLoader()


def load_knowledge_extraction_prompt(
    text_content: str, domain: str = "universal"  # Domain-agnostic default
) -> str:
    """Convenience function to load knowledge extraction prompt"""
    return prompt_loader.render_knowledge_extraction_prompt(text_content, domain)


def get_available_prompt_templates() -> list:
    """Convenience function to get available templates"""
    return prompt_loader.list_available_templates()
