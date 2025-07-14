#!/usr/bin/env python3
"""
CLI wrapper for the Comprehensive GNN Trainer.

- Runs the full research-level GNN training pipeline (hyperparameter optimization, ablation, cross-validation, evaluation).
- Accepts a JSON config file or uses the default config.
- Allows override of Optuna trial count and cross-validation folds.

Usage:
  python scripts/train_comprehensive_gnn.py --config scripts/example_comprehensive_gnn_config.json --n_trials 10 --k_folds 3
  python scripts/train_comprehensive_gnn.py  # uses default config

See scripts/example_comprehensive_gnn_config.json for config structure.
"""
import argparse
import sys
from src.gnn.comprehensive_trainer import run_comprehensive_gnn_training, create_comprehensive_training_config

def main():
    parser = argparse.ArgumentParser(description="Comprehensive GNN Training CLI")
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file (optional)')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials (default: 100)')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of cross-validation folds (default: 5)')
    args = parser.parse_args()

    # Load or create config
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_comprehensive_training_config()

    # Override trials/folds if specified
    config['n_trials'] = args.n_trials
    config['k_folds'] = args.k_folds

    print(f"[INFO] Starting comprehensive GNN training with config: {config}")
    results = run_comprehensive_gnn_training()
    print("[INFO] Training complete. Results saved.")

if __name__ == "__main__":
    main()