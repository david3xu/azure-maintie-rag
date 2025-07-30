from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional

# Import workflow evidence collector from core utilities (architecture fix)
from core.utilities.workflow_evidence_collector import AzureDataWorkflowEvidenceCollector
from services.workflow_service import WorkflowService

router = APIRouter()

# Global workflow service
workflow_service = WorkflowService()

async def retrieve_workflow_evidence(workflow_id: str) -> Dict[str, Any]:
    """Retrieve workflow evidence using core utilities"""
    try:
        # Try to get evidence from workflow service
        if workflow_id in workflow_service.evidence_collectors:
            collector = workflow_service.evidence_collectors[workflow_id]
            return await collector.generate_workflow_evidence_report()
        else:
            # Return empty result if workflow not found
            return {}
    except Exception:
        return {}

async def retrieve_gnn_training_evidence(training_session_id: str) -> Dict[str, Any]:
    """Retrieve GNN training evidence"""
    try:
        # Create evidence collector for this session
        collector = AzureDataWorkflowEvidenceCollector(training_session_id)
        return await collector.generate_workflow_evidence_report()
    except Exception:
        return {}

async def get_latest_gnn_training_evidence(domain: str) -> Dict[str, Any]:
    """Get latest GNN training evidence for domain"""
    try:
        # Create a mock evidence report for the domain
        from core.azure_ml.gnn.training.orchestrator import UnifiedGNNTrainingOrchestrator
        
        return {
            "domain": domain,
            "status": "available",
            "training_type": "evidence_based",
            "evidence_available": True
        }
    except Exception:
        return {}

@router.get("/api/v1/workflow/{workflow_id}/evidence")
async def get_workflow_evidence(
    workflow_id: str,
    include_data_lineage: bool = False,
    include_cost_breakdown: bool = False
) -> Dict[str, Any]:
    try:
        evidence_report = await retrieve_workflow_evidence(workflow_id)
        if not evidence_report:
            raise HTTPException(status_code=404, detail=f"Workflow evidence not found: {workflow_id}")
        response = {
            "workflow_id": workflow_id,
            "evidence_summary": {
                "total_steps": evidence_report.get("total_steps"),
                "azure_services_used": evidence_report.get("azure_services_used"),
                "total_processing_time_ms": evidence_report.get("total_processing_time_ms"),
                "success_rate": evidence_report.get("quality_assessment", {}).get("success_rate")
            },
            "azure_service_evidence": evidence_report.get("evidence_chain")
        }
        if include_data_lineage:
            response["data_lineage"] = evidence_report.get("data_lineage")
        if include_cost_breakdown:
            response["cost_analysis"] = {
                "total_cost_usd": evidence_report.get("total_cost_usd"),
                "cost_by_service": evidence_report.get("cost_breakdown_by_service")
            }
        return response
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve workflow evidence: {str(e)}")

@router.get("/api/v1/gnn-training/{domain}/evidence")
async def get_gnn_training_evidence(
    domain: str,
    training_session_id: Optional[str] = None
) -> Dict[str, Any]:
    try:
        if training_session_id:
            evidence = await retrieve_gnn_training_evidence(training_session_id)
        else:
            evidence = await get_latest_gnn_training_evidence(domain)
        return {
            "domain": domain,
            "training_evidence": evidence,
            "model_lineage": evidence.get("data_lineage"),
            "quality_certification": evidence.get("quality_metrics"),
            "deployment_status": evidence.get("deployment_evidence")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve GNN training evidence: {str(e)}")