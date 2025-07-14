# Comprehensive GNN Trainer: CLI & Config Guide

This directory contains scripts and configuration for running the research-level GNN training pipeline in MaintIE RAG.

## CLI Script
- **File:** `train_comprehensive_gnn.py`
- **Purpose:** Run the full GNN training pipeline (hyperparameter optimization, ablation, cross-validation, evaluation) from the command line.
- **Usage:**
  ```bash
  python scripts/train_comprehensive_gnn.py --config scripts/example_comprehensive_gnn_config.json --n_trials 10 --k_folds 3
  python scripts/train_comprehensive_gnn.py  # uses default config
  ```
- **Options:**
  - `--config`: Path to a JSON config file (see below)
  - `--n_trials`: Number of Optuna trials (default: 100)
  - `--k_folds`: Number of cross-validation folds (default: 5)

## Example Config
- **File:** `example_comprehensive_gnn_config.json`
- **Purpose:** Template for configuring the trainer. Edit values as needed for your experiment.
- **Structure:**
  - `experiment_name`: Name for experiment tracking
  - `use_wandb`: Enable Weights & Biases logging
  - `model_config`: Model architecture parameters
  - `optimizer`, `learning_rate`, `weight_decay`, etc.: Training hyperparameters
  - `max_epochs`, `patience`, etc.: Training control
  - `batch_size`, `num_workers`: Data loading
  - `gradient_clip`, `noise_std`: Regularization

## Integration
- The CLI and config are used in CI/CD for smoke testing.
- See `src/gnn/comprehensive_trainer.py` for full pipeline details.

## See Also
- [src/gnn/comprehensive_trainer.py](../src/gnn/comprehensive_trainer.py) for API usage and documentation.