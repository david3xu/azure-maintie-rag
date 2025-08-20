#!/usr/bin/env python3
"""
Path Utilities for Dataflow Scripts
===================================

Provides consistent path resolution for dataflow scripts regardless of
where they are called from (make targets, direct execution, etc.)
"""

from pathlib import Path


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    # This utility is in scripts/dataflow/utilities/, so go up 4 levels to get to project root
    # utilities -> dataflow -> scripts -> azure-maintie-rag
    potential_root = Path(__file__).parent.parent.parent.parent

    # Handle container environments where the path might be /workspace/azure-maintie-rag
    # Check if the resolved path exists and has expected structure
    if potential_root.exists() and (potential_root / "azure.yaml").exists():
        return potential_root

    # Fallback: Look for azure.yaml file by walking up the directory tree
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "azure.yaml").exists():
            return parent

    # Last resort: return the calculated path and let mkdir create what's needed
    return potential_root


def get_results_dir() -> Path:
    """Get the absolute path to the results directory, creating it if needed."""
    project_root = get_project_root()
    results_dir = project_root / "scripts" / "dataflow" / "results"

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        print(f"⚠️  Warning: Could not create results directory {results_dir}: {e}")
        print(f"🔍 Project root resolved to: {project_root}")
        print(f"🔍 Results dir path: {results_dir}")
        # Try to create without parents first
        try:
            results_dir.mkdir(exist_ok=True)
        except (OSError, PermissionError):
            print(f"❌ Failed to create results directory. Using temp directory.")
            import tempfile
            temp_results = Path(tempfile.mkdtemp(prefix="dataflow_results_"))
            print(f"📁 Using temporary results directory: {temp_results}")
            return temp_results

    return results_dir


def get_results_file(filename: str) -> Path:
    """Get the absolute path to a specific results file."""
    return get_results_dir() / filename


def get_data_dir() -> Path:
    """Get the absolute path to the data directory."""
    project_root = get_project_root()
    return project_root / "data"


def get_raw_data_dir() -> Path:
    """Get the absolute path to the raw data directory."""
    return get_data_dir() / "raw"
