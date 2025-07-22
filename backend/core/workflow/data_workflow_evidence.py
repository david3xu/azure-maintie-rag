from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from config.settings import azure_settings
from core.workflow.data_workflow_evidence import AzureServiceCostTracker

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

class AzureDataWorkflowEvidenceCollector:
    """Collect and correlate evidence across Azure services"""
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.evidence_chain: List[DataWorkflowEvidence] = []
        self.azure_correlation_ids: Dict[str, str] = {}

    async def record_azure_service_evidence(
        self,
        step_number: int,
        azure_service: str,
        operation_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        processing_time_ms: float,
        azure_request_id: str
    ) -> None:
        evidence = DataWorkflowEvidence(
            workflow_id=self.workflow_id,
            step_number=step_number,
            azure_service=azure_service,
            operation_type=operation_type,
            input_data=input_data,
            output_data=output_data,
            processing_time_ms=processing_time_ms,
            cost_estimate_usd=await self._calculate_service_cost(azure_service, input_data),
            quality_metrics=await self._assess_output_quality(output_data),
            timestamp=datetime.now().isoformat()
        )
        self.evidence_chain.append(evidence)
        self.azure_correlation_ids[f"step_{step_number}"] = azure_request_id

    async def generate_workflow_evidence_report(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "total_steps": len(self.evidence_chain),
            "total_processing_time_ms": sum(e.processing_time_ms for e in self.evidence_chain),
            "total_cost_usd": sum(e.cost_estimate_usd or 0 for e in self.evidence_chain),
            "azure_services_used": list(set(e.azure_service for e in self.evidence_chain)),
            "evidence_chain": [e.__dict__ for e in self.evidence_chain],
            "azure_correlation_map": self.azure_correlation_ids,
            "data_lineage": await self._build_data_lineage(),
            "quality_assessment": await self._aggregate_quality_metrics()
        }

    async def _calculate_service_cost(self, azure_service: str, input_data: Dict[str, Any]) -> float:
        cost_tracker = AzureServiceCostTracker()
        usage = {}
        if azure_service == "blob_storage":
            usage = {"documents": len(input_data.get("docs", []))}
        elif azure_service == "openai":
            usage = {"tokens": input_data.get("token_count", 0)}
        elif azure_service == "cognitive_search":
            usage = {"documents": input_data.get("doc_count", 0)}
        elif azure_service == "cosmos_db":
            usage = {"operations": 1}
        return cost_tracker.calculate_workflow_cost({azure_service: usage}).get(azure_service, 0.0)

    async def _assess_output_quality(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        # Simple real quality check: success flag and count fields
        quality = {"success": output_data.get("success", False)}
        if "doc_count" in output_data:
            quality["doc_count"] = output_data["doc_count"]
        if "accuracy" in output_data:
            quality["accuracy"] = output_data["accuracy"]
        return quality

    async def _build_data_lineage(self) -> Dict[str, Any]:
        # Build lineage as a list of step outputs (e.g., URIs, IDs)
        return {f"step_{i+1}": e.output_data.get("uri") or e.output_data.get("id") for i, e in enumerate(self.evidence_chain) if e.output_data.get("uri") or e.output_data.get("id")}

    async def _aggregate_quality_metrics(self) -> Dict[str, Any]:
        # Aggregate quality metrics from all steps
        metrics = {}
        for e in self.evidence_chain:
            for k, v in e.quality_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
        # For numeric metrics, compute average
        agg = {}
        for k, v in metrics.items():
            if all(isinstance(x, (int, float)) for x in v):
                agg[k] = sum(v) / len(v)
            else:
                agg[k] = v
        return agg