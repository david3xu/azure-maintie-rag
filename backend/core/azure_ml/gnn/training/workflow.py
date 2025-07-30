#!/usr/bin/env python3
"""
Azure ML GNN Training Workflow
Enterprise training script for Azure ML compute clusters
"""
import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from trainer import UniversalGNNTrainer, UniversalGNNConfig
from data_loader import load_graph_data
# from model_quality_assessor import GNNModelQualityAssessor  # To be implemented

def main():
    parser = argparse.ArgumentParser(description="Azure ML GNN Training Workflow")
    parser.add_argument("--domain", required=True, help="Domain for training")
    parser.add_argument("--graph_data", required=True, help="Graph data blob path")
    parser.add_argument("--output_path", default="./data/outputs", help="Output path for trained model")
    args = parser.parse_args()

    # Azure ML environment integration
    import mlflow
    mlflow.start_run()

    try:
        # Load graph data from Azure Blob Storage
        train_loader, val_loader = load_graph_data(args.graph_data, args.domain)
        if not train_loader.dataset:
            raise ValueError("No training data available")
        # Training configuration (data-driven from environment)
        config = UniversalGNNConfig(
            hidden_dim=int(os.getenv("GNN_HIDDEN_DIM", "64")),
            num_layers=int(os.getenv("GNN_NUM_LAYERS", "3")),
            learning_rate=float(os.getenv("GNN_LEARNING_RATE", "0.01")),
            epochs=int(os.getenv("GNN_NUM_EPOCHS", "100"))
        )
        # Model training
        trainer = UniversalGNNTrainer(config)
        # Get model dimensions from data
        sample_batch = next(iter(train_loader))
        num_features = sample_batch.x.size(1)
        num_classes = len(torch.unique(sample_batch.y))
        trainer.setup_model(num_features, num_classes)
        # Execute training
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.epochs,
            save_path=f"{args.output_path}/model.pt"
        )
        # Model quality assessment
        quality_metrics = assess_model_quality(trainer.model, val_loader, args.domain, training_results)
        # Log metrics to Azure ML
        mlflow.log_metrics(training_results["final_metrics"])
        mlflow.log_metrics(quality_metrics)
        mlflow.log_param("domain", args.domain)
        mlflow.log_param("config", config.__dict__)
        # Save model to Azure ML
        mlflow.pytorch.log_model(trainer.model, "gnn_model")
        print(f"Training completed.")
    except Exception as e:
        mlflow.log_param("error", str(e))
        raise
    finally:
        mlflow.end_run()

def assess_model_quality(model, val_loader, domain: str, training_results: Dict[str, Any]) -> Dict[str, float]:
    """Assess GNN model quality with domain-specific metrics"""
    model.eval()
    
    # Basic performance metrics
    final_metrics = training_results.get("final_metrics", {})
    val_accuracy = final_metrics.get("val_accuracy", 0.0)
    val_loss = final_metrics.get("val_loss", 1.0)
    
    # Model complexity metrics
    num_parameters = sum(p.numel() for p in model.parameters())
    model_size_mb = num_parameters * 4 / (1024 * 1024)  # Approximate size in MB
    
    # Training stability metrics
    training_history = training_results.get("training_history", [])
    loss_variance = torch.var(torch.tensor([h["val_loss"] for h in training_history])).item() if training_history else 0.0
    
    # Domain-specific quality thresholds
    domain_thresholds = {
        "maintenance": {"min_accuracy": 0.75, "max_loss": 0.5},
        "general": {"min_accuracy": 0.70, "max_loss": 0.6}
    }
    threshold = domain_thresholds.get(domain, domain_thresholds["general"])
    
    # Quality assessment
    quality_metrics = {
        "model_quality_score": min(val_accuracy / threshold["min_accuracy"], 1.0),
        "performance_stability": 1.0 - min(loss_variance, 1.0),
        "model_efficiency": max(0.0, 1.0 - (model_size_mb / 100.0)),  # Penalize models > 100MB
        "meets_threshold": val_accuracy >= threshold["min_accuracy"] and val_loss <= threshold["max_loss"],
        "validation_accuracy": val_accuracy,
        "validation_loss": val_loss,
        "model_size_mb": model_size_mb,
        "parameter_count": num_parameters
    }
    
    # Overall quality score
    quality_metrics["overall_quality"] = (
        quality_metrics["model_quality_score"] * 0.5 +
        quality_metrics["performance_stability"] * 0.3 +
        quality_metrics["model_efficiency"] * 0.2
    )
    
    return quality_metrics

if __name__ == "__main__":
    main()