"""
Centralized Data Models for Azure Universal RAG Agents
=====================================================

This module now serves as a compatibility layer that imports from the
modular data models structure. The models have been reorganized into
logical modules for better maintainability and separation of concerns.

**Modular Structure:**
- agents.core.models.base: Foundation models and PydanticAI integration
- agents.core.models.azure: Azure service models and configurations
- agents.core.models.agents: Agent contracts and dependency models
- agents.core.models.workflow: Workflow state and execution models
- agents.core.models.search: Search request/response models
- agents.core.models.extraction: Knowledge extraction models
- agents.core.models.validation: Validation and error models
- agents.core.models.cache: Cache and performance models

This approach provides:
- Clear separation of concerns
- Easier maintenance and updates
- Better code organization
- Backward compatibility with existing imports
"""

# Import all models from the modular structure for backward compatibility
from .models import *

# Re-export module metadata
__version__ = "2.0.0"
__description__ = "Modular data models for Azure Universal RAG system"

# Maintain backward compatibility by ensuring all previously exported models are available
# The __all__ list is inherited from the models.__init__ module
