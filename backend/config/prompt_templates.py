"""
Configurable prompt templates for different query types
"""

from typing import Dict
from pathlib import Path
from config.advanced_settings import advanced_settings


class ConfigurablePromptTemplates:
    """Manage prompt templates with configuration"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from configuration or files"""

        # Default templates (can be overridden by config files)
        default_templates = {
            "troubleshooting": """
You are helping with a maintenance troubleshooting issue. Please provide a comprehensive troubleshooting response for the following:

Query: {query}
Equipment Category: {equipment_category}
Identified Entities: {entities}
Related Concepts: {expanded_concepts}
Urgency Level: {urgency}

Relevant Documentation:
{context}

Safety Considerations:
{safety_considerations}

Please provide:
1. A systematic troubleshooting approach
2. Most likely causes ranked by probability
3. Step-by-step diagnostic procedures
4. Required tools and materials
5. Safety precautions specific to this equipment
6. When to escalate to specialized technicians

Format your response with clear headings and actionable steps.
""",

            "procedural": """
You are providing maintenance procedure guidance. Please create a detailed procedural response for:

Query: {query}
Equipment Category: {equipment_category}
Identified Entities: {entities}
Related Concepts: {expanded_concepts}

Relevant Documentation:
{context}

Safety Considerations:
{safety_considerations}

Please provide:
1. Step-by-step procedure with clear instructions
2. Required tools, parts, and materials
3. Safety precautions and PPE requirements
4. Quality checks and verification steps
5. Common pitfalls to avoid
6. Estimated time and skill level required

Use numbered steps and include safety reminders throughout.
""",

            "preventive": """
You are providing preventive maintenance guidance. Please create a comprehensive preventive maintenance response for:

Query: {query}
Equipment Category: {equipment_category}
Identified Entities: {entities}
Related Concepts: {expanded_concepts}

Relevant Documentation:
{context}

Safety Considerations:
{safety_considerations}

Please provide:
1. Recommended maintenance schedule and frequency
2. Inspection points and criteria
3. Lubrication requirements
4. Parts replacement intervals
5. Performance monitoring parameters
6. Documentation and record-keeping requirements

Focus on preventing failures and optimizing equipment life.
""",

            "safety": """
You are providing safety-focused maintenance guidance. Please create a safety-centered response for:

Query: {query}
Equipment Category: {equipment_category}
Identified Entities: {entities}
Related Concepts: {expanded_concepts}

Relevant Documentation:
{context}

Safety Considerations:
{safety_considerations}

Please provide:
1. Comprehensive safety assessment
2. Required safety procedures and protocols
3. Personal protective equipment (PPE) requirements
4. Hazard identification and mitigation
5. Emergency procedures and contacts
6. Regulatory compliance requirements

Prioritize safety above all other considerations.
""",

            "general": """
You are providing general maintenance guidance. Please create a helpful response for:

Query: {query}
Equipment Category: {equipment_category}
Identified Entities: {entities}
Related Concepts: {expanded_concepts}

Relevant Documentation:
{context}

Safety Considerations:
{safety_considerations}

Please provide accurate, practical maintenance guidance that addresses the query comprehensively. Include relevant safety considerations and cite the provided documentation where applicable.
"""
        }

        # TODO: Add ability to load templates from external files
        # templates_dir = Path("config/templates")
        # if templates_dir.exists():
        #     for template_file in templates_dir.glob("*.txt"):
        #         template_name = template_file.stem
        #         with open(template_file, 'r') as f:
        #             default_templates[template_name] = f.read()

        return default_templates

    def get_template(self, template_type: str) -> str:
        """Get template by type"""
        return self.templates.get(template_type, self.templates["general"])

    def update_template(self, template_type: str, template_content: str) -> None:
        """Update template dynamically"""
        self.templates[template_type] = template_content

    def get_all_templates(self) -> Dict[str, str]:
        """Get all available templates"""
        return self.templates.copy()

    def add_template(self, template_type: str, template_content: str) -> None:
        """Add a new template"""
        self.templates[template_type] = template_content


# Global template manager
template_manager = ConfigurablePromptTemplates()
