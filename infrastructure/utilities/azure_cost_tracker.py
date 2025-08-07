"""
Azure Service Cost Tracking Utilities
Moved from services layer to fix core->services dependency violation.
Provides cost estimation for Azure service operations in workflows.
"""

import logging
from typing import Any, Dict

from infrastructure.constants import AzurePricingConstants

logger = logging.getLogger(__name__)


class AzureServiceCostTracker:
    """Azure service cost correlation for workflow transparency"""

    def __init__(self):
        self.cost_per_service = {
            "azure_openai": {
                "per_token": AzurePricingConstants.AZURE_OPENAI_PER_TOKEN, 
                "per_request": AzurePricingConstants.AZURE_OPENAI_PER_REQUEST
            },
            "cognitive_search": {
                "per_document": AzurePricingConstants.COGNITIVE_SEARCH_PER_DOCUMENT, 
                "per_query": AzurePricingConstants.COGNITIVE_SEARCH_PER_QUERY
            },
            "cosmos_db": {
                "per_operation": AzurePricingConstants.COSMOS_DB_PER_OPERATION, 
                "per_ru": AzurePricingConstants.COSMOS_DB_PER_RU
            },
            "blob_storage": {
                "per_gb_month": AzurePricingConstants.BLOB_STORAGE_PER_GB_MONTH, 
                "per_operation": AzurePricingConstants.BLOB_STORAGE_PER_OPERATION
            },
            "azure_ml": {
                "per_training_hour": AzurePricingConstants.AZURE_ML_PER_TRAINING_HOUR, 
                "per_inference": AzurePricingConstants.AZURE_ML_PER_INFERENCE
            },
        }

    def _calculate_service_cost(self, service: str, usage: dict) -> float:
        """Calculate cost for a specific service based on usage"""
        cost = AzurePricingConstants.ZERO_COST
        rates = self.cost_per_service.get(service, {})

        for key, value in usage.items():
            rate_key = f"per_{key}"
            if rate_key in rates:
                cost += rates[rate_key] * value

        return cost

    def calculate_workflow_cost(self, service_usage: dict) -> dict:
        """Calculate total workflow cost across all services"""
        return {
            service: self._calculate_service_cost(service, usage)
            for service, usage in service_usage.items()
        }

    def estimate_gnn_training_cost(
        self, domain: str, data_size_gb: float, training_hours: float
    ) -> Dict[str, float]:
        """Estimate cost for GNN training workflow"""
        costs = {}

        # Azure ML training cost
        costs["azure_ml_training"] = (
            training_hours * self.cost_per_service["azure_ml"]["per_training_hour"]
        )

        # Cosmos DB operations (graph queries)
        estimated_graph_operations = int(data_size_gb * 1000)  # Rough estimate
        costs["cosmos_db_operations"] = self._calculate_service_cost(
            "cosmos_db",
            {
                "operation": estimated_graph_operations,
                "ru": estimated_graph_operations * 5,  # Average RUs per operation
            },
        )

        # Storage cost for training data
        costs["blob_storage"] = self._calculate_service_cost(
            "blob_storage",
            {"gb_month": data_size_gb, "operation": 100},  # Read/write operations
        )

        # Total cost
        costs["total_estimated"] = sum(costs.values())

        logger.info(
            f"Estimated GNN training cost for {domain}: ${costs['total_estimated']:.4f}"
        )
        return costs

    def get_service_rates(self, service: str) -> Dict[str, float]:
        """Get cost rates for a specific service"""
        return self.cost_per_service.get(service, {})

    def add_custom_rate(self, service: str, rate_type: str, rate: float):
        """Add or update a custom rate for a service"""
        if service not in self.cost_per_service:
            self.cost_per_service[service] = {}

        self.cost_per_service[service][f"per_{rate_type}"] = rate
        logger.info(f"Updated {service} rate for {rate_type}: ${rate}")

    def estimate_operation_cost(
        self, service: str, operation_type: str, usage_count: int
    ) -> float:
        """Estimate cost for a specific operation"""
        usage = {operation_type: usage_count}
        return self._calculate_service_cost(service, usage)
