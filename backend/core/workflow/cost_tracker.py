class AzureServiceCostTracker:
    """Azure service cost correlation for workflow transparency"""
    def __init__(self):
        self.cost_per_service = {
            "azure_openai": {"per_token": 0.00002, "per_request": 0.001},
            "cognitive_search": {"per_document": 0.01, "per_query": 0.005},
            "cosmos_db": {"per_operation": 0.0001, "per_ru": 0.00008},
            "blob_storage": {"per_gb_month": 0.018, "per_operation": 0.0001}
        }
    def _calculate_service_cost(self, service: str, usage: dict) -> float:
        cost = 0.0
        rates = self.cost_per_service.get(service, {})
        for key, value in usage.items():
            rate_key = f"per_{key}"
            if rate_key in rates:
                cost += rates[rate_key] * value
        return cost
    def calculate_workflow_cost(self, service_usage: dict) -> dict:
        return {
            service: self._calculate_service_cost(service, usage)
            for service, usage in service_usage.items()
        }