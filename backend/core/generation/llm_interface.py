"""
LLM Interface - Universal Import
Imports universal components only
"""

from core.generation.universal_llm_interface import (
    UniversalLLMInterface,
    create_universal_llm_interface
)

# Export only universal interface
__all__ = [
    'UniversalLLMInterface',
    'create_universal_llm_interface'
]