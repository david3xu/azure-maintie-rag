"""
Workflow Evidence Collection Utilities
Moved from services layer to fix core->services dependency violation.
Provides evidence tracking for Azure service operations in GNN training workflows.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import azure_settings

logger = logging.getLogger(__name__)


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
    azure_request_id: Optional[str] = None


class AzureDataWorkflowEvidenceCollector:
    """Collect and correlate evidence across Azure services"""

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.evidence_chain: List[DataWorkflowEvidence] = []
        self.azure_correlation_ids: Dict[str, str] = {}
        # Import cost tracker from utilities
        from .azure_cost_tracker import AzureServiceCostTracker

        self.cost_tracker = AzureServiceCostTracker()

    async def record_azure_service_evidence(
        self,
        step_number: int,
        azure_service: str,
        operation_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        processing_time_ms: float,
        azure_request_id: Optional[str] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> DataWorkflowEvidence:
        """Record evidence for an Azure service operation"""

        # Calculate cost estimate
        usage_data = self._extract_usage_from_operation(
            azure_service, input_data, output_data
        )
        cost_estimate = self.cost_tracker._calculate_service_cost(
            azure_service, usage_data
        )

        evidence = DataWorkflowEvidence(
            workflow_id=self.workflow_id,
            step_number=step_number,
            azure_service=azure_service,
            operation_type=operation_type,
            input_data=input_data,
            output_data=output_data,
            processing_time_ms=processing_time_ms,
            cost_estimate_usd=cost_estimate,
            quality_metrics=quality_metrics or {},
            timestamp=datetime.now().isoformat(),
            azure_request_id=azure_request_id,
        )

        self.evidence_chain.append(evidence)

        # Store correlation ID if provided
        if azure_request_id:
            self.azure_correlation_ids[f"step_{step_number}"] = azure_request_id

        logger.debug(
            f"Recorded evidence for {azure_service} operation: {operation_type}"
        )
        return evidence

    def _extract_usage_from_operation(
        self, service: str, input_data: Dict, output_data: Dict
    ) -> Dict:
        """Extract usage metrics from operation data"""
        usage = {}

        if service == "azure_openai":
            usage["token"] = input_data.get("token_count", 0) + output_data.get(
                "token_count", 0
            )
            usage["request"] = 1
        elif service == "cognitive_search":
            usage["document"] = len(input_data.get("documents", []))
            usage["query"] = input_data.get("query_count", 1)
        elif service == "cosmos_db":
            usage["operation"] = 1
            usage["ru"] = output_data.get("ru_charge", 5)  # Default RU estimate
        elif service == "blob_storage":
            usage["operation"] = 1

        return usage

    async def generate_workflow_evidence_report(self) -> Dict[str, Any]:
        """Generate comprehensive workflow evidence report"""
        try:
            if not self.evidence_chain:
                return {
                    "workflow_id": self.workflow_id,
                    "status": "no_evidence",
                    "message": "No evidence collected for this workflow",
                }

            # Calculate summary metrics
            total_cost = sum(e.cost_estimate_usd or 0.0 for e in self.evidence_chain)
            total_processing_time = sum(
                e.processing_time_ms for e in self.evidence_chain
            )

            # Group by service
            service_breakdown = {}
            for evidence in self.evidence_chain:
                service = evidence.azure_service
                if service not in service_breakdown:
                    service_breakdown[service] = {
                        "operations": 0,
                        "total_cost": 0.0,
                        "total_time_ms": 0.0,
                    }

                service_breakdown[service]["operations"] += 1
                service_breakdown[service]["total_cost"] += (
                    evidence.cost_estimate_usd or 0.0
                )
                service_breakdown[service][
                    "total_time_ms"
                ] += evidence.processing_time_ms

            # Generate report
            report = {
                "workflow_id": self.workflow_id,
                "report_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_steps": len(self.evidence_chain),
                    "total_cost_usd": total_cost,
                    "total_processing_time_ms": total_processing_time,
                    "services_used": list(service_breakdown.keys()),
                },
                "service_breakdown": service_breakdown,
                "evidence_chain": [
                    {
                        "step": e.step_number,
                        "service": e.azure_service,
                        "operation": e.operation_type,
                        "processing_time_ms": e.processing_time_ms,
                        "cost_usd": e.cost_estimate_usd,
                        "timestamp": e.timestamp,
                        "azure_request_id": e.azure_request_id,
                    }
                    for e in self.evidence_chain
                ],
                "correlation_ids": self.azure_correlation_ids,
            }

            logger.info(f"Generated evidence report for workflow {self.workflow_id}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate workflow evidence report: {e}")
            return {"workflow_id": self.workflow_id, "status": "error", "error": str(e)}

    def get_evidence_by_service(self, service: str) -> List[DataWorkflowEvidence]:
        """Get all evidence for a specific Azure service"""
        return [e for e in self.evidence_chain if e.azure_service == service]

    def get_evidence_by_step(self, step_number: int) -> Optional[DataWorkflowEvidence]:
        """Get evidence for a specific step"""
        for evidence in self.evidence_chain:
            if evidence.step_number == step_number:
                return evidence
        return None

    def get_total_cost(self) -> float:
        """Get total estimated cost for all operations"""
        return sum(e.cost_estimate_usd or 0.0 for e in self.evidence_chain)

    def get_total_processing_time(self) -> float:
        """Get total processing time in milliseconds"""
        return sum(e.processing_time_ms for e in self.evidence_chain)
