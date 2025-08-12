"""Consolidated utilities for Universal RAG system.

This package contains all utilities used across the Universal RAG system,
consolidated from the original utilities/ directory.
"""

from .azure_cost_tracker import AzureServiceCostTracker

# Removed PromptTemplateLoader - using unified approach with Agent 1 template variables only
from .workflow_evidence_collector import (
    AzureDataWorkflowEvidenceCollector,
    DataWorkflowEvidence,
)

__all__ = [
    "AzureDataWorkflowEvidenceCollector",
    "DataWorkflowEvidence",
    "AzureServiceCostTracker",
    "PromptTemplateLoader",
]
