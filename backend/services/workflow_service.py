"""
Workflow Service - Business logic for workflow management and tracking
Consolidated from core/workflow/ modules: progress_tracker, cost_tracker, data_workflow_evidence, azure-workflow-manager
"""

import logging
import time
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
import threading
from dataclasses import dataclass, asdict
from enum import Enum

from config.settings import azure_settings
from core.utilities.workflow_evidence_collector import AzureDataWorkflowEvidenceCollector, DataWorkflowEvidence
from core.utilities.azure_cost_tracker import AzureServiceCostTracker

logger = logging.getLogger(__name__)


# ===== WORKFLOW STEP DEFINITIONS (from progress_tracker.py) =====

class WorkflowStep(Enum):
    """Workflow step enumeration for progress tracking"""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    BLOB_STORAGE = "blob_storage"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    SEARCH_INDEXING = "search_indexing"
    COSMOS_STORAGE = "cosmos_storage"
    VALIDATION = "validation"
    COMPLETION = "completion"


class AzureServiceType(Enum):
    """Azure service types for workflow management"""
    OPENAI = "azure_openai"
    SEARCH = "cognitive_search"
    COSMOS = "cosmos_db"
    STORAGE = "blob_storage"


# ===== DATA MODELS =====

@dataclass
class ProgressStatus:
    """Progress status tracking for workflow steps"""
    current_step: WorkflowStep
    percentage: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    

@dataclass
class DataWorkflowEvidence:
    """Enterprise data workflow evidence tracking"""
    workflow_id: str
    step_number: int
    azure_service: str
    operation_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time_ms: float
    cost_estimate_usd: Optional[float]
    quality_metrics: Dict[str, Any]
    timestamp: str


# ===== PROGRESS TRACKING SERVICE =====

class WorkflowProgressTracker:
    """Real-time progress tracker for Azure workflows"""
    
    def __init__(self):
        self.current_workflows: Dict[str, Dict[str, Any]] = {}
        self.step_percentages = {
            WorkflowStep.INITIALIZATION: 10,
            WorkflowStep.DATA_LOADING: 20,
            WorkflowStep.BLOB_STORAGE: 35,
            WorkflowStep.KNOWLEDGE_EXTRACTION: 50,
            WorkflowStep.SEARCH_INDEXING: 70,
            WorkflowStep.COSMOS_STORAGE: 85,
            WorkflowStep.VALIDATION: 95,
            WorkflowStep.COMPLETION: 100
        }
    
    def start_workflow(self, workflow_id: str, total_steps: int = 8) -> Dict[str, Any]:
        """Start tracking a new workflow"""
        workflow_data = {
            "id": workflow_id,
            "status": "running",
            "current_step": WorkflowStep.INITIALIZATION,
            "percentage": 0,
            "total_steps": total_steps,
            "start_time": datetime.now(),
            "last_update": datetime.now(),
            "steps_completed": [],
            "current_message": "Initializing workflow..."
        }
        
        self.current_workflows[workflow_id] = workflow_data
        return workflow_data
    
    def update_progress(
        self, 
        workflow_id: str, 
        step: WorkflowStep, 
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update workflow progress"""
        
        if workflow_id not in self.current_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.current_workflows[workflow_id]
        workflow["current_step"] = step
        workflow["percentage"] = self.step_percentages.get(step, 0)
        workflow["current_message"] = message
        workflow["last_update"] = datetime.now()
        
        if details:
            workflow["details"] = details
        
        # Track completed steps
        if step not in workflow["steps_completed"]:
            workflow["steps_completed"].append(step)
        
        return workflow
    
    def complete_workflow(self, workflow_id: str, final_message: str = "Workflow completed successfully") -> Dict[str, Any]:
        """Mark workflow as completed"""
        if workflow_id not in self.current_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.current_workflows[workflow_id]
        workflow["status"] = "completed"
        workflow["current_step"] = WorkflowStep.COMPLETION
        workflow["percentage"] = 100
        workflow["current_message"] = final_message
        workflow["end_time"] = datetime.now()
        
        return workflow
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        return self.current_workflows.get(workflow_id)


# ===== MAIN WORKFLOW SERVICE =====

class WorkflowService:
    """
    Unified Workflow Service
    Provides workflow management, progress tracking, evidence collection, and cost monitoring
    """
    
    def __init__(self):
        self.progress_tracker = WorkflowProgressTracker()
        self.evidence_collectors: Dict[str, AzureDataWorkflowEvidenceCollector] = {}
        self.cost_tracker = AzureServiceCostTracker()
        
        logger.info("WorkflowService initialized")
    
    def create_workflow(self, workflow_name: str = None) -> str:
        """Create a new workflow and return workflow ID"""
        workflow_id = workflow_name or f"workflow_{uuid.uuid4().hex[:8]}"
        
        # Initialize progress tracking
        self.progress_tracker.start_workflow(workflow_id)
        
        # Initialize evidence collection
        self.evidence_collectors[workflow_id] = AzureDataWorkflowEvidenceCollector(workflow_id)
        
        logger.info(f"Created workflow: {workflow_id}")
        return workflow_id
    
    async def execute_full_pipeline(self, source_data_path: str, domain: str) -> Dict[str, Any]:
        """Execute complete intelligent RAG pipeline: extraction → graph → GNN → query"""
        try:
            workflow_id = self.create_workflow(f"full_pipeline_{domain}")
            start_time = datetime.now()
            
            # Import services for pipeline execution
            from services.data_service import DataService
            from services.knowledge_service import KnowledgeService
            from services.ml_service import MLService
            from services.infrastructure_service import InfrastructureService
            
            infrastructure = InfrastructureService()
            data_service = DataService(infrastructure)
            knowledge_service = KnowledgeService()
            ml_service = MLService()
            
            pipeline_results = {
                "workflow_id": workflow_id,
                "domain": domain,
                "source_path": source_data_path,
                "start_time": start_time.isoformat(),
                "stages": {},
                "success": False
            }
            
            # Stage 1: Knowledge Extraction
            self.update_workflow_progress(workflow_id, WorkflowStep.KNOWLEDGE_EXTRACTION, "Starting LLM knowledge extraction")
            extraction_result = await knowledge_service.extract_from_file(source_data_path, domain)
            pipeline_results["stages"]["knowledge_extraction"] = extraction_result
            
            if not extraction_result.get('success', False):
                pipeline_results["error"] = "Knowledge extraction failed"
                return pipeline_results
            
            # Stage 2: Data Migration (using extracted knowledge)
            self.update_workflow_progress(workflow_id, WorkflowStep.DATA_LOADING, "Migrating data to Azure services")
            migration_result = await data_service.migrate_data_to_azure(source_data_path, domain)
            pipeline_results["stages"]["data_migration"] = migration_result
            
            # Stage 3: GNN Training (if graph data available)
            if migration_result.get("migrations", {}).get("cosmos", {}).get("success", False):
                self.update_workflow_progress(workflow_id, WorkflowStep.COMPLETION, "Training GNN model")
                gnn_result = await ml_service.train_gnn_model(domain)
                pipeline_results["stages"]["gnn_training"] = gnn_result
            
            # Determine overall success
            extraction_success = extraction_result.get('success', False)
            migration_success = migration_result.get('status') in ['completed', 'functional_degraded']
            
            pipeline_results["success"] = extraction_success and migration_success
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["duration"] = str(datetime.now() - start_time)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Full pipeline execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id if 'workflow_id' in locals() else None
            }
    
    def update_workflow_progress(
        self, 
        workflow_id: str, 
        step: WorkflowStep, 
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update workflow progress with step information"""
        return self.progress_tracker.update_progress(workflow_id, step, message, details)
    
    async def record_azure_operation(
        self,
        workflow_id: str,
        step_number: int,
        azure_service: str,
        operation_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        processing_time_ms: float,
        quality_metrics: Optional[Dict[str, Any]] = None
    ) -> DataWorkflowEvidence:
        """Record evidence of an Azure service operation"""
        
        if workflow_id not in self.evidence_collectors:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        collector = self.evidence_collectors[workflow_id]
        return await collector.record_azure_service_evidence(
            step_number, azure_service, operation_type,
            input_data, output_data, processing_time_ms, quality_metrics
        )
    
    def calculate_workflow_costs(self, workflow_id: str) -> Dict[str, Any]:
        """Calculate total costs for a workflow"""
        if workflow_id not in self.evidence_collectors:
            return {"error": f"Workflow {workflow_id} not found"}
        
        collector = self.evidence_collectors[workflow_id]
        
        # Aggregate usage by service
        service_usage = {}
        total_cost = 0.0
        
        for evidence in collector.evidence_chain:
            service = evidence.azure_service
            cost = evidence.cost_estimate_usd or 0.0
            
            if service not in service_usage:
                service_usage[service] = {"operations": 0, "total_cost": 0.0}
            
            service_usage[service]["operations"] += 1
            service_usage[service]["total_cost"] += cost
            total_cost += cost
        
        return {
            "workflow_id": workflow_id,
            "total_cost_usd": total_cost,
            "service_breakdown": service_usage,
            "evidence_count": len(collector.evidence_chain)
        }
    
    def get_workflow_evidence(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get all evidence for a workflow"""
        if workflow_id not in self.evidence_collectors:
            return []
        
        collector = self.evidence_collectors[workflow_id]
        return [asdict(evidence) for evidence in collector.evidence_chain]
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive workflow status"""
        progress_status = self.progress_tracker.get_workflow_status(workflow_id)
        
        if not progress_status:
            return None
        
        # Add cost and evidence information
        costs = self.calculate_workflow_costs(workflow_id)
        evidence_count = len(self.get_workflow_evidence(workflow_id))
        
        return {
            **progress_status,
            "cost_analysis": costs,
            "evidence_items": evidence_count
        }
    
    def complete_workflow(self, workflow_id: str, final_message: str = "Workflow completed successfully") -> Dict[str, Any]:
        """Complete a workflow and return final status"""
        final_status = self.progress_tracker.complete_workflow(workflow_id, final_message)
        
        # Add final cost analysis
        final_costs = self.calculate_workflow_costs(workflow_id)
        final_status["final_cost_analysis"] = final_costs
        
        logger.info(f"Completed workflow {workflow_id} with total cost: ${final_costs.get('total_cost_usd', 0.0):.4f}")
        
        return final_status
    
    async def initialize_rag_orchestration(self, domain_name: str = "general", 
                                         text_files: Optional[List] = None, 
                                         force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Initialize RAG system orchestration
        Coordinates knowledge extraction, indexing, and graph construction
        """
        workflow_id = self.create_workflow("rag_orchestration", {
            "domain": domain_name,
            "force_rebuild": force_rebuild,
            "text_files_count": len(text_files) if text_files else 0
        })
        
        try:
            # Start workflow
            self.update_progress(workflow_id, WorkflowStep.INITIALIZATION, 0.1, 
                               f"Initializing RAG system for domain: {domain_name}")
            
            # If no text files provided, discover them
            if not text_files:
                from pathlib import Path
                raw_data_path = Path(azure_settings.raw_data_dir)
                text_files = []
                for pattern in azure_settings.raw_data_include_patterns:
                    text_files.extend(raw_data_path.glob(pattern))
            
            if not text_files:
                self.complete_workflow(workflow_id, "No text files found for processing")
                return {"success": False, "error": "No text files found"}
            
            # Update progress
            self.update_progress(workflow_id, WorkflowStep.DATA_LOADING, 0.2,
                               f"Found {len(text_files)} files for processing")
            
            # Process through data service
            from services.data_service import DataService
            data_service = DataService(self.infrastructure)
            
            # Migrate data to Azure
            self.update_progress(workflow_id, WorkflowStep.BLOB_STORAGE, 0.3,
                               "Uploading data to Azure Blob Storage")
            
            migration_result = await data_service.migrate_data_to_azure(
                str(text_files[0].parent) if text_files else "data/raw", 
                domain_name
            )
            
            if not migration_result.get("success", False):
                self.complete_workflow(workflow_id, f"Migration failed: {migration_result.get('error', 'Unknown error')}")
                return migration_result
            
            # Track evidence
            self.add_workflow_evidence(
                workflow_id, 
                step_number=1,
                azure_service="blob_storage",
                operation_type="data_migration",
                input_data={"files_count": len(text_files)},
                output_data=migration_result,
                processing_time_ms=migration_result.get("duration_seconds", 0) * 1000
            )
            
            # Complete workflow
            self.complete_workflow(workflow_id, "RAG orchestration completed successfully")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "domain": domain_name,
                "files_processed": len(text_files),
                "migration_results": migration_result,
                "workflow_status": self.get_workflow_status(workflow_id)
            }
            
        except Exception as e:
            logger.error(f"RAG orchestration failed: {str(e)}")
            self.complete_workflow(workflow_id, f"Failed with error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id
            }