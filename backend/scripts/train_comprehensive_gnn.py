"""Azure ML control script for Universal GNN training."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.run import Run

from backend.core.azure_ml.gnn.trainer import train_gnn_with_azure_ml
from backend.core.models.universal_rag_models import UniversalTrainingConfig

logger = logging.getLogger(__name__)


def run_comprehensive_gnn_training(config_path: Optional[str] = None,
                                 workspace_name: Optional[str] = None,
                                 experiment_name: str = "universal-rag-gnn") -> Dict[str, Any]:
    """
    Run GNN training with Azure ML

    Args:
        config_path: Path to configuration file
        workspace_name: Azure ML workspace name
        experiment_name: Experiment name

    Returns:
        Training results
    """
    try:
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "model_type": "gnn",
                "domain": "general",
                "epochs": 100,
                "patience": 20,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.5,
                "conv_type": "gcn",
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "batch_size": 32
            }

        logger.info(f"Using configuration: {config}")

        # Check if running in Azure ML
        run = Run.get_context()
        if run.id.startswith('OfflineRun'):
            # Running locally
            logger.info("Running GNN training locally")
            return run_local_training(config)
        else:
            # Running in Azure ML
            logger.info("Running GNN training in Azure ML")
            return run_azure_ml_training(config, workspace_name, experiment_name)

    except Exception as e:
        logger.error(f"GNN training failed: {e}")
        return {"error": str(e)}


def run_local_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run GNN training locally

    Args:
        config: Training configuration

    Returns:
        Training results
    """
    try:
        # Create output directory
        output_dir = Path("models/gnn")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"gnn_model_{config.get('domain', 'general')}.pt"

        # Run training
        result = train_gnn_with_azure_ml(
            config_dict=config,
            data_path=config.get("data_path", "backend/data/"),
            output_path=str(output_path)
        )

        return {
            "success": True,
            "model_path": str(output_path),
            "training_result": result.to_dict() if result else None
        }

    except Exception as e:
        logger.error(f"Local training failed: {e}")
        return {"error": str(e)}


def run_azure_ml_training(config: Dict[str, Any],
                         workspace_name: Optional[str],
                         experiment_name: str) -> Dict[str, Any]:
    """
    Run GNN training in Azure ML

    Args:
        config: Training configuration
        workspace_name: Azure ML workspace name
        experiment_name: Experiment name

    Returns:
        Training results
    """
    try:
        # Load workspace
        if workspace_name:
            ws = Workspace.from_config()
        else:
            ws = Workspace.from_config()

        # Create experiment
        experiment = Experiment(workspace=ws, name=experiment_name)

        # Create script config
        script_config = ScriptRunConfig(
            source_directory='./backend/core/azure_ml/gnn',
            script='trainer.py',
            compute_target=config.get('compute_target', 'cpu-cluster'),
            arguments=[
                '--config', json.dumps(config),
                '--output_path', 'models/gnn_model.pt'
            ]
        )

        # Set up environment
        env = Environment.from_conda_specification(
            name='gnn-env',
            file_path='gnn-env.yml'
        )
        script_config.run_config.environment = env

        # Submit run
        run = experiment.submit(script_config)

        logger.info(f"Submitted Azure ML run: {run.id}")

        return {
            "success": True,
            "run_id": run.id,
            "experiment_name": experiment_name,
            "status": "submitted"
        }

    except Exception as e:
        logger.error(f"Azure ML training failed: {e}")
        return {"error": str(e)}


def create_gnn_environment_file():
    """Create conda environment file for GNN training"""
    env_content = """name: gnn-env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pytorch
  - torch-geometric
  - torch-scatter
  - torch-sparse
  - torch-cluster
  - torch-spline-conv
  - numpy
  - scipy
  - scikit-learn
  - matplotlib
  - pandas
  - azureml-core
  - azure-identity
  - azure-storage-blob
  - azure-search-documents
  - azure-cosmos
  - gremlinpython
  - pip
  - pip:
    - azure-ai-ml
    - azure-core
    - azure-storage-blob
    - azure-search-documents
    - azure-cosmos
    - gremlinpython
    - torch-geometric
    - optuna
    - wandb
"""

    with open('gnn-env.yml', 'w') as f:
        f.write(env_content)

    logger.info("Created gnn-env.yml environment file")


def create_example_config():
    """Create example configuration file"""
    config = {
        "model_type": "gnn",
        "domain": "general",
        "epochs": 100,
        "patience": 20,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "conv_type": "gcn",
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "compute_target": "cpu-cluster",
        "data_path": "backend/data/",
        "output_path": "models/gnn_model.pt"
    }

    with open('example_comprehensive_gnn_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Created example_comprehensive_gnn_config.json")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Universal GNN with Azure ML")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--workspace', type=str, help='Azure ML workspace name')
    parser.add_argument('--experiment', type=str, default='universal-rag-gnn',
                       help='Experiment name')
    parser.add_argument('--create-env', action='store_true',
                       help='Create conda environment file')
    parser.add_argument('--create-config', action='store_true',
                       help='Create example configuration file')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create environment file if requested
    if args.create_env:
        create_gnn_environment_file()
        return

    # Create config file if requested
    if args.create_config:
        create_example_config()
        return

    # Run training
    result = run_comprehensive_gnn_training(
        config_path=args.config,
        workspace_name=args.workspace,
        experiment_name=args.experiment
    )

    if result.get("success"):
        logger.info("Training completed successfully")
        logger.info(f"Result: {result}")
    else:
        logger.error(f"Training failed: {result.get('error')}")


if __name__ == "__main__":
    main()