"""Azure ML service integrations for Universal RAG"""

from .classification_client import (
    AzureClassificationPipeline as AzureMLClassificationService,
)

# GNN components with optional PyTorch imports
# Use lazy loading to avoid PyTorch resource exhaustion during initial imports
_gnn_components = None
_gnn_available_components = [
    "GNNTrainingClient", "GNNInferenceClient", "UniversalGNN", 
    "UniversalGNNConfig", "create_gnn_model"
]

def _load_gnn_components():
    """Load GNN components only when first accessed."""
    global _gnn_components
    if _gnn_components is None:
        try:
            from .gnn_inference_client import GNNInferenceClient
            from .gnn_model import UniversalGNN, UniversalGNNConfig, create_gnn_model
            from .gnn_training_client import GNNTrainingClient
            _gnn_components = {
                "GNNTrainingClient": GNNTrainingClient,
                "GNNInferenceClient": GNNInferenceClient,
                "UniversalGNN": UniversalGNN,
                "UniversalGNNConfig": UniversalGNNConfig,
                "create_gnn_model": create_gnn_model,
            }
        except ImportError:
            # Mark as attempted but failed
            _gnn_components = {}
    return _gnn_components

# Set up __all__ to include all possible exports (lazy-loaded)
__all__ = ["AzureMLClassificationService"] + _gnn_available_components

# Make lazy-loaded components available as module attributes
def __getattr__(name):
    if name in _gnn_available_components:
        components = _load_gnn_components()
        if name in components:
            return components[name]
        else:
            raise ImportError(f"GNN component '{name}' not available (PyTorch dependency issue)")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
