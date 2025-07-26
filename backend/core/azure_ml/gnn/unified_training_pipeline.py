"""
Unified GNN Training Pipeline.

This module provides a single entry point for the complete extraction → training
workflow, consolidating the multiple competing orchestrators identified in the
design analysis.

Key Features:
- End-to-end pipeline from extraction file to trained model
- Integrated quality validation and feature engineering
- Comprehensive error handling and logging
- Support for both local and Azure ML training
- Parallel processing where possible

Created as part of GNN Training Stage Design Analysis remediation plan.
Location: /docs/workflows/GNN_Training_Stage_Design_Analysis.md
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from datetime import datetime

from core.models.gnn_data_models import (
    StandardizedGraphData,
    GNNTrainingConfig,
    TrainingResult
)
from .data_bridge import ExtractionToGNNBridge, GraphDataValidator
from .feature_engineering import SemanticFeatureEngine, FeaturePipeline
from .trainer import UniversalGNNTrainer
from .model import UniversalGNNConfig
from core.azure_openai.completion_service import AzureOpenAIService

logger = logging.getLogger(__name__)


class UnifiedGNNTrainingPipeline:
    """
    Single entry point for extraction → training workflow.
    
    This class consolidates the fragmented orchestration layer into a
    streamlined pipeline that handles the complete workflow from
    knowledge extraction output to trained GNN model.
    """
    
    def __init__(
        self,
        openai_service: Optional[AzureOpenAIService] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize unified training pipeline.
        
        Args:
            openai_service: Azure OpenAI service for semantic embeddings
            cache_dir: Directory for caching embeddings and models
            device: Device for training ("cpu", "cuda", or auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.data_bridge = ExtractionToGNNBridge()
        self.validator = GraphDataValidator()
        self.semantic_engine = SemanticFeatureEngine(
            openai_service=openai_service,
            cache_dir=cache_dir
        )
        self.feature_pipeline = FeaturePipeline(
            semantic_engine=self.semantic_engine,
            normalize_features=True
        )
        
        logger.info(f"UnifiedGNNTrainingPipeline initialized on device: {self.device}")
    
    async def train_from_extraction(
        self,
        extraction_file: Union[str, Path],
        config: Optional[GNNTrainingConfig] = None,
        domain: Optional[str] = None,
        output_dir: Optional[str] = None,
        validate_quality: bool = True
    ) -> TrainingResult:
        """
        Complete end-to-end training from extraction file.
        
        This is the main entry point that handles the entire pipeline:
        1. Convert extraction to standardized format
        2. Validate graph quality  
        3. Generate semantic features
        4. Train GNN model
        5. Save and validate results
        
        Args:
            extraction_file: Path to knowledge extraction JSON file
            config: Training configuration (uses defaults if not provided)
            domain: Domain name (inferred if not provided)
            output_dir: Output directory for model and artifacts
            validate_quality: Whether to validate graph quality before training
            
        Returns:
            TrainingResult with metrics and model path
            
        Example:
            >>> pipeline = UnifiedGNNTrainingPipeline(openai_service)
            >>> result = await pipeline.train_from_extraction(
            ...     "backend/data/extraction_outputs/clean_knowledge_extraction.json",
            ...     domain="maintenance"
            ... )
            >>> if result.success:
            ...     print(f"Model saved to: {result.model_path}")
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting end-to-end GNN training from {extraction_file}")
            
            # Default configuration
            if config is None:
                config = GNNTrainingConfig()
            
            # Setup output directory
            if output_dir is None:
                output_dir = "backend/models/gnn"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Convert extraction to standardized format
            logger.info("Step 1: Converting extraction to standardized format")
            graph_data = self.data_bridge.convert_extraction_to_gnn_data(
                extraction_file=extraction_file,
                domain=domain
            )
            
            # Step 2: Validate graph quality
            if validate_quality:
                logger.info("Step 2: Validating graph quality")
                validation_report = self.validator.comprehensive_validation(graph_data)
                
                if not validation_report["overall_valid"]:
                    error_msg = f"Graph quality validation failed: {validation_report['recommendations']}"
                    logger.error(error_msg)
                    return TrainingResult(
                        success=False,
                        error_message=error_msg,
                        domain=graph_data.domain,
                        config_used=config.to_dict()
                    )
                
                logger.info("Graph quality validation passed")
            
            # Step 3: Generate semantic features
            logger.info("Step 3: Generating semantic features")
            node_features, edge_features, edge_indices = await self.feature_pipeline.process_graph_data(graph_data)
            
            # Step 4: Create PyTorch Geometric data
            logger.info("Step 4: Creating PyTorch Geometric data")
            pytorch_data = self._create_pytorch_data(
                node_features, edge_features, edge_indices, graph_data
            )
            
            # Step 5: Train GNN model
            logger.info("Step 5: Training GNN model")
            training_result = await self._train_gnn_model(
                pytorch_data, config, graph_data.domain, output_path
            )
            
            # Step 6: Save training artifacts
            logger.info("Step 6: Saving training artifacts")
            self._save_training_artifacts(
                graph_data, config, training_result, output_path
            )
            
            training_time = time.time() - start_time
            
            # Create final result
            result = TrainingResult(
                success=training_result["success"],
                model_path=training_result.get("model_path"),
                training_metrics=training_result.get("training_metrics"),
                validation_metrics=training_result.get("validation_metrics"),
                training_time_seconds=training_time,
                model_version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                domain=graph_data.domain,
                config_used=config.to_dict(),
                error_message=training_result.get("error_message")
            )
            
            if result.success:
                logger.info(f"Training completed successfully in {training_time:.1f}s")
                logger.info(f"Model saved to: {result.model_path}")
            else:
                logger.error(f"Training failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Pipeline failed after {training_time:.1f}s: {e}")
            
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time_seconds=training_time,
                domain=domain,
                config_used=config.to_dict() if config else None
            )
    
    def preview_extraction(
        self,
        extraction_file: Union[str, Path],
        max_entities: int = 10,
        max_relations: int = 10
    ) -> Dict[str, Any]:
        """
        Preview extraction conversion without full processing.
        
        Useful for debugging and validation before full training.
        
        Args:
            extraction_file: Path to extraction file
            max_entities: Maximum entities to show
            max_relations: Maximum relations to show
            
        Returns:
            Preview information with statistics and samples
        """
        return self.data_bridge.preview_conversion(
            extraction_file, max_entities, max_relations
        )
    
    def validate_extraction_quality(
        self,
        extraction_file: Union[str, Path],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate extraction quality without training.
        
        Args:
            extraction_file: Path to extraction file
            domain: Domain name (inferred if not provided)
            
        Returns:
            Comprehensive validation report
        """
        try:
            # Convert extraction
            graph_data = self.data_bridge.convert_extraction_to_gnn_data(
                extraction_file=extraction_file,
                domain=domain
            )
            
            # Validate quality
            validation_report = self.validator.comprehensive_validation(graph_data)
            validation_report["conversion_success"] = True
            
            return validation_report
            
        except Exception as e:
            return {
                "conversion_success": False,
                "error": str(e),
                "overall_valid": False
            }
    
    def _create_pytorch_data(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        edge_indices: np.ndarray,
        graph_data: StandardizedGraphData
    ) -> Data:
        """
        Create PyTorch Geometric Data object.
        
        Args:
            node_features: Node feature matrix
            edge_features: Edge feature matrix  
            edge_indices: Edge index matrix
            graph_data: Original graph data
            
        Returns:
            PyTorch Geometric Data object
        """
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        
        if edge_features.size > 0:
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_attr = None
        
        # Create simple node labels for training (entity type classification)
        self.semantic_engine.fit_type_encoders(graph_data)
        entity_type_encoder = self.semantic_engine.entity_type_encoder
        
        node_labels = []
        for entity in graph_data.entities:
            type_id = entity_type_encoder.type_to_id.get(entity.entity_type, entity_type_encoder.unknown_id)
            node_labels.append(type_id)
        
        y = torch.tensor(node_labels, dtype=torch.long)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_classes=len(entity_type_encoder.type_to_id)
        )
        
        logger.info(f"Created PyTorch data: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_classes} classes")
        return data
    
    async def _train_gnn_model(
        self,
        pytorch_data: Data,
        config: GNNTrainingConfig,
        domain: str,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Train GNN model with the prepared data.
        
        Args:
            pytorch_data: PyTorch Geometric data
            config: Training configuration
            domain: Domain name
            output_path: Output directory
            
        Returns:
            Training results
        """
        try:
            # Convert to UniversalGNNConfig
            gnn_config = UniversalGNNConfig(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                conv_type=config.model_type
            )
            
            # Initialize trainer
            trainer = UniversalGNNTrainer(gnn_config, device=self.device)
            
            # Setup model
            num_node_features = pytorch_data.x.size(1)
            num_classes = pytorch_data.num_classes
            
            trainer.setup_model(num_node_features, num_classes)
            
            # Create data loaders
            train_loader = DataLoader([pytorch_data], batch_size=1, shuffle=False)
            val_loader = DataLoader([pytorch_data], batch_size=1, shuffle=False)  # Using same data for validation
            
            # Train model
            model_path = output_path / f"gnn_model_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            
            training_results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.epochs,
                patience=config.patience,
                save_path=str(model_path)
            )
            
            # Extract metrics
            final_metrics = training_results["training_history"][-1] if training_results["training_history"] else {}
            
            return {
                "success": True,
                "model_path": str(model_path),
                "training_metrics": {
                    "final_train_loss": final_metrics.get("train_loss", 0.0),
                    "final_train_acc": final_metrics.get("train_acc", 0.0),
                    "epochs_trained": training_results.get("final_epoch", 0),
                    "early_stopped": training_results.get("early_stopped", False)
                },
                "validation_metrics": {
                    "final_val_loss": final_metrics.get("val_loss", 0.0),
                    "final_val_acc": final_metrics.get("val_acc", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {
                "success": False,
                "error_message": str(e)
            }
    
    def _save_training_artifacts(
        self,
        graph_data: StandardizedGraphData,
        config: GNNTrainingConfig,
        training_result: Dict[str, Any],
        output_path: Path
    ):
        """
        Save training artifacts for reproducibility and analysis.
        
        Args:
            graph_data: Original graph data
            config: Training configuration
            training_result: Training results
            output_path: Output directory
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save graph data
            graph_file = output_path / f"graph_data_{graph_data.domain}_{timestamp}.json"
            graph_data.save_to_file(graph_file)
            
            # Save configuration
            config_file = output_path / f"training_config_{graph_data.domain}_{timestamp}.json"
            with open(config_file, 'w') as f:
                import json
                json.dump(config.to_dict(), f, indent=2)
            
            # Save training results
            results_file = output_path / f"training_results_{graph_data.domain}_{timestamp}.json"
            with open(results_file, 'w') as f:
                import json
                json.dump(training_result, f, indent=2, default=str)
            
            logger.info(f"Saved training artifacts to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training artifacts: {e}")


class BatchTrainingPipeline:
    """
    Batch training pipeline for processing multiple extraction files.
    
    Useful for training on multiple domains or large datasets.
    """
    
    def __init__(self, unified_pipeline: UnifiedGNNTrainingPipeline):
        """
        Initialize batch training pipeline.
        
        Args:
            unified_pipeline: Unified pipeline instance
        """
        self.pipeline = unified_pipeline
    
    async def train_batch(
        self,
        extraction_files: List[Union[str, Path]],
        config: Optional[GNNTrainingConfig] = None,
        output_dir: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[TrainingResult]:
        """
        Train models on multiple extraction files concurrently.
        
        Args:
            extraction_files: List of extraction file paths
            config: Training configuration
            output_dir: Output directory
            max_concurrent: Maximum concurrent training jobs
            
        Returns:
            List of training results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def train_single(file_path):
            async with semaphore:
                return await self.pipeline.train_from_extraction(
                    extraction_file=file_path,
                    config=config,
                    output_dir=output_dir
                )
        
        # Execute batch training
        tasks = [train_single(file_path) for file_path in extraction_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(TrainingResult(
                    success=False,
                    error_message=str(result),
                    config_used=config.to_dict() if config else None
                ))
            else:
                final_results.append(result)
        
        # Log batch summary
        successful = sum(1 for r in final_results if r.success)
        logger.info(f"Batch training completed: {successful}/{len(extraction_files)} successful")
        
        return final_results
    
    def generate_batch_report(
        self,
        results: List[TrainingResult],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive batch training report.
        
        Args:
            results: List of training results
            output_file: Optional file to save report
            
        Returns:
            Batch training report
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        report = {
            "batch_summary": {
                "total_jobs": len(results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": len(successful_results) / len(results) if results else 0.0
            },
            "performance_metrics": {},
            "error_analysis": {},
            "recommendations": []
        }
        
        # Performance metrics
        if successful_results:
            training_times = [r.training_time_seconds for r in successful_results if r.training_time_seconds]
            if training_times:
                report["performance_metrics"] = {
                    "avg_training_time": np.mean(training_times),
                    "min_training_time": np.min(training_times),
                    "max_training_time": np.max(training_times)
                }
        
        # Error analysis
        if failed_results:
            error_counts = {}
            for result in failed_results:
                error_type = type(result.error_message).__name__ if result.error_message else "Unknown"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            report["error_analysis"] = {
                "error_types": error_counts,
                "sample_errors": [r.error_message for r in failed_results[:5]]
            }
        
        # Generate recommendations
        if report["batch_summary"]["success_rate"] < 0.8:
            report["recommendations"].append("Success rate below 80% - review extraction quality")
        
        if report["performance_metrics"].get("avg_training_time", 0) > 300:  # 5 minutes
            report["recommendations"].append("Average training time high - consider model optimization")
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
        
        return report