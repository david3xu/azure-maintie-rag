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
    return Path(__file__).parent.parent.parent.parent


def get_results_dir() -> Path:
    """Get the absolute path to the results directory, creating it if needed."""
    project_root = get_project_root()
    results_dir = project_root / "scripts" / "dataflow" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
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