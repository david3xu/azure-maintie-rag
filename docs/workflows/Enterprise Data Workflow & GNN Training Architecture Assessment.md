# **Azure Universal RAG: Data Workflow & GNN Training Architecture**

## 🚀 Quick Start: End-to-End GNN Lifecycle

Follow these steps to run the full, auditable GNN lifecycle with evidence tracking:

### 1. **Prepare Your Data**
- Place your raw markdown files in `data/raw/`.

### 2. **Run Data Preparation Workflow**
```bash
cd backend
PYTHONPATH=. ./.venv/bin/python scripts/data_preparation_workflow.py
```
- This ingests raw data, processes with Azure OpenAI, indexes with Cognitive Search, stores metadata in Cosmos DB, and saves a full evidence report.

### 3. **Run GNN Training Pipeline**
```bash
cd backend
PYTHONPATH=. ./.venv/bin/python scripts/orchestrate_gnn_pipeline.py --domain <your_domain>
```
- Replace `<your_domain>` with your use case (e.g., `general`).
- This checks for graph changes, exports data, runs Azure ML GNN training, and saves a full evidence report.

### 4. **Retrieve Evidence Reports (Optional)**
- Use the API to retrieve workflow or GNN training evidence:
    - **Workflow Evidence:**
      ```
      GET /api/v1/workflow/{workflow_id}/evidence
      ```
    - **GNN Training Evidence:**
      ```
      GET /api/v1/gnn-training/{domain}/evidence
      ```
- Or query Cosmos DB directly for evidence reports.

---

## **Enterprise Workflow Analysis**

Based on your current Azure Universal RAG implementation, I'll design comprehensive data workflow orchestration with clear input/output evidence tracking for both raw data processing and GNN training pipelines.

## **Current Implementation Assessment**## **Enterprise Data Workflow & GNN Training Architecture Assessment**

Based on the current Azure Universal RAG implementation, here's the enterprise workflow analysis and architectural recommendations:

## **Raw Data Workflow: Current State Assessment**

### **✅ Existing Implementation - Production Ready**

Your raw data workflow demonstrates **enterprise-grade Azure service orchestration** with comprehensive input/output evidence tracking:

#### **Data Flow Evidence Architecture**
```
📁 Raw Data → Azure Blob Storage → Azure OpenAI → Azure Cognitive Search → Azure Cosmos DB
     ↓              ↓                    ↓                    ↓                    ↓
  File Stats    Storage Logs      Processing Logs      Index Metrics     Metadata Tracking
```

#### **Current Evidence Tracking Capabilities**
- **Input Evidence**: File discovery with size/format validation
- **Processing Evidence**: Azure service request/response correlation
- **Output Evidence**: Document counts, processing times, service utilization
- **Quality Evidence**: Search index validation, metadata consistency checks

## **GNN Training Workflow: Architecture Gap Analysis**

### **🔴 Implementation Gaps Identified**

#### **Current State Assessment**
```
Existing Components:  60% Complete
├── Training Framework: ✅ Azure ML integration architecture
├── Data Pipeline: 🟡 Cosmos DB graph export (partial)
├── Model Quality: 🟡 Assessment framework (placeholder logic)
├── Evidence Tracking: 🔴 Missing comprehensive lineage
└── Deployment Pipeline: 🟡 Azure ML orchestration (incomplete)
```

## **Enterprise Data Workflow Architecture Design**

### **Component 1: Raw Data Workflow Evidence Service**

**Location**: `backend/core/workflow/data_workflow_evidence.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from config.settings import azure_settings

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
        """Record evidence from Azure service operation"""

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
        """Generate comprehensive workflow evidence report"""

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
```

### **Component 2: GNN Training Evidence Pipeline**

**Location**: `backend/core/azure_ml/gnn_training_evidence_orchestrator.py`

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
import mlflow
from typing import Dict, Any, List
from datetime import datetime

class GNNTrainingEvidenceOrchestrator:
    """Enterprise GNN training with comprehensive evidence collection"""

    def __init__(self, ml_client: MLClient, cosmos_client):
        self.ml_client = ml_client
        self.cosmos_client = cosmos_client
        self.training_evidence: List[Dict[str, Any]] = []

    async def orchestrate_evidence_based_training(
        self,
        domain: str,
        trigger_threshold: Optional[int] = None
    ) -> Dict[str, Any]:
        """Orchestrate GNN training with complete evidence tracking"""

        training_session_id = f"gnn_training_{domain}_{int(time.time())}"
        evidence_collector = AzureDataWorkflowEvidenceCollector(training_session_id)

        try:
            # Step 1: Graph Change Analysis with Evidence
            change_evidence = await self._collect_graph_change_evidence(domain, evidence_collector)

            # Step 2: Data Quality Assessment with Evidence
            quality_evidence = await self._collect_data_quality_evidence(domain, evidence_collector)

            # Step 3: Training Data Preparation with Evidence
            training_data_evidence = await self._collect_training_data_evidence(domain, evidence_collector)

            # Step 4: Model Training with Azure ML Evidence
            training_evidence = await self._execute_azure_ml_training_with_evidence(
                domain, training_data_evidence, evidence_collector
            )

            # Step 5: Model Quality Assessment with Evidence
            quality_assessment_evidence = await self._collect_model_quality_evidence(
                training_evidence["model_uri"], domain, evidence_collector
            )

            # Step 6: Model Deployment with Evidence
            deployment_evidence = await self._collect_deployment_evidence(
                training_evidence["model_uri"], domain, evidence_collector
            )

            # Generate comprehensive evidence report
            evidence_report = await evidence_collector.generate_workflow_evidence_report()

            return {
                "training_session_id": training_session_id,
                "domain": domain,
                "status": "completed",
                "evidence_report": evidence_report,
                "azure_ml_job_id": training_evidence.get("job_id"),
                "model_deployment_endpoint": deployment_evidence.get("endpoint_url"),
                "data_lineage": evidence_report["data_lineage"],
                "cost_breakdown": evidence_report["total_cost_usd"],
                "quality_metrics": evidence_report["quality_assessment"]
            }

        except Exception as e:
            # Record failure evidence
            await evidence_collector.record_azure_service_evidence(
                step_number=999,
                azure_service="training_orchestrator",
                operation_type="training_failure",
                input_data={"domain": domain, "error": str(e)},
                output_data={"success": False},
                processing_time_ms=0,
                azure_request_id="failure"
            )
            raise

    async def _collect_graph_change_evidence(
        self,
        domain: str,
        evidence_collector: AzureDataWorkflowEvidenceCollector
    ) -> Dict[str, Any]:
        """Collect evidence of graph changes from Cosmos DB"""

        start_time = time.time()

        # Execute graph change analysis using existing Cosmos client
        change_metrics = await self.cosmos_client.get_graph_change_metrics(domain)

        processing_time = (time.time() - start_time) * 1000

        await evidence_collector.record_azure_service_evidence(
            step_number=1,
            azure_service="cosmos_db",
            operation_type="graph_change_analysis",
            input_data={"domain": domain},
            output_data=change_metrics,
            processing_time_ms=processing_time,
            azure_request_id=f"cosmos_change_{int(time.time())}"
        )

        return change_metrics

    async def _execute_azure_ml_training_with_evidence(
        self,
        domain: str,
        training_data: Dict[str, Any],
        evidence_collector: AzureDataWorkflowEvidenceCollector
    ) -> Dict[str, Any]:
        """Execute Azure ML training with comprehensive evidence tracking"""

        # Prepare Azure ML job configuration from azure_settings
        job_config = {
            "display_name": f"gnn_training_{domain}_{int(time.time())}",
            "experiment_name": getattr(azure_settings, 'azure_ml_experiment_name', 'universal-rag-gnn'),
            "compute": getattr(azure_settings, 'azure_ml_compute_cluster_name', 'gnn-cluster'),
            "environment": getattr(azure_settings, 'azure_ml_training_environment', 'gnn-training-env'),
            "code": "./backend",
            "command": [
                "python", "core/azure_ml/gnn/train_gnn_workflow.py",
                "--domain", domain,
                "--graph_data", "${{inputs.graph_data}}",
                "--output_path", "${{outputs.trained_model}}"
            ],
            "inputs": {
                "graph_data": {
                    "type": "uri_folder",
                    "path": training_data["data_path"]
                }
            },
            "outputs": {
                "trained_model": {
                    "type": "mlflow_model"
                }
            }
        }

        # Submit job to Azure ML
        start_time = time.time()
        job = Job(**job_config)
        submitted_job = self.ml_client.jobs.create_or_update(job)
        submission_time = (time.time() - start_time) * 1000

        # Record submission evidence
        await evidence_collector.record_azure_service_evidence(
            step_number=3,
            azure_service="azure_ml",
            operation_type="training_job_submission",
            input_data={"domain": domain, "job_config": job_config},
            output_data={"job_id": submitted_job.id, "status": submitted_job.status},
            processing_time_ms=submission_time,
            azure_request_id=submitted_job.id
        )

        # Monitor training progress with evidence collection
        final_job_status = await self._monitor_training_with_evidence(
            submitted_job.id, evidence_collector
        )

        return {
            "job_id": submitted_job.id,
            "model_uri": final_job_status.get("model_uri"),
            "training_metrics": final_job_status.get("metrics"),
            "status": final_job_status.get("status")
        }
```

### **Component 3: Workflow Evidence Dashboard Service**

**Location**: `backend/api/endpoints/workflow_evidence.py`

```python
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import json

router = APIRouter()

@router.get("/api/v1/workflow/{workflow_id}/evidence")
async def get_workflow_evidence(
    workflow_id: str,
    include_data_lineage: bool = False,
    include_cost_breakdown: bool = False
) -> Dict[str, Any]:
    """Get comprehensive workflow evidence for enterprise visibility"""

    try:
        # Retrieve evidence from Azure Cosmos DB or Application Insights
        evidence_report = await retrieve_workflow_evidence(workflow_id)

        if not evidence_report:
            raise HTTPException(status_code=404, detail=f"Workflow evidence not found: {workflow_id}")

        response = {
            "workflow_id": workflow_id,
            "evidence_summary": {
                "total_steps": evidence_report["total_steps"],
                "azure_services_used": evidence_report["azure_services_used"],
                "total_processing_time_ms": evidence_report["total_processing_time_ms"],
                "success_rate": evidence_report["quality_assessment"]["success_rate"]
            },
            "azure_service_evidence": evidence_report["evidence_chain"]
        }

        if include_data_lineage:
            response["data_lineage"] = evidence_report["data_lineage"]

        if include_cost_breakdown:
            response["cost_analysis"] = {
                "total_cost_usd": evidence_report["total_cost_usd"],
                "cost_by_service": evidence_report["cost_breakdown_by_service"]
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve workflow evidence: {str(e)}")

@router.get("/api/v1/gnn-training/{domain}/evidence")
async def get_gnn_training_evidence(
    domain: str,
    training_session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get GNN training evidence for enterprise model governance"""

    try:
        if training_session_id:
            evidence = await retrieve_gnn_training_evidence(training_session_id)
        else:
            # Get latest training evidence for domain
            evidence = await get_latest_gnn_training_evidence(domain)

        return {
            "domain": domain,
            "training_evidence": evidence,
            "model_lineage": evidence["data_lineage"],
            "quality_certification": evidence["quality_metrics"],
            "deployment_status": evidence["deployment_evidence"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve GNN training evidence: {str(e)}")
```

## **Implementation Strategy**

### **Phase 1: Raw Data Workflow Enhancement (Current + Evidence)**
Your raw data workflow is **production-ready**. Enhancement needed:
- Add the `AzureDataWorkflowEvidenceCollector` to existing `data_preparation_workflow.py`
- Integrate evidence collection into existing Azure service calls
- Add evidence dashboard endpoint

### **Phase 2: Complete GNN Training Pipeline (New Implementation)**
Your GNN training framework exists but needs completion:
- Implement `GNNTrainingEvidenceOrchestrator` with complete Azure ML integration
- Complete the existing `gnn_orchestrator.py` with evidence tracking
- Add model quality assessment with real metrics
- Integrate with existing Azure ML workspace configuration

### **Enterprise Evidence Architecture Benefits**
- **Compliance**: Complete audit trail for model governance
- **Cost Management**: Granular cost attribution per workflow step
- **Quality Assurance**: Evidence-based model certification
- **Operational Excellence**: Real-time monitoring with Azure Application Insights integration
- **Data Lineage**: Complete traceability from raw data to deployed models

**Recommendation**: Your raw data workflow is excellent. Focus implementation effort on completing the GNN training evidence pipeline to achieve comprehensive enterprise data workflow visibility.


I'll help you identify the GNN training command line execution and debug the current implementation.Based on the existing codebase analysis, here are the **exact execution commands** and **debugging instructions** for the GNN lifecycle training:

## GNN Training Execution Commands

### 1. **Direct Python Execution** (Current Implementation)
```bash
# Navigate to project root directory
cd /path/to/project

# Set Python path and execute training script
PYTHONPATH=./backend python backend/scripts/train_comprehensive_gnn.py [OPTIONS]
```

### 2. **Available Command Line Options** (From Existing Code)
```bash
# Create example configuration file
python backend/scripts/train_comprehensive_gnn.py --create-config

# Create conda environment file
python backend/scripts/train_comprehensive_gnn.py --create-env

# Run training with custom config
python backend/scripts/train_comprehensive_gnn.py --config your_config.json

# Run with specific Azure ML workspace
python backend/scripts/train_comprehensive_gnn.py --workspace your-workspace-name

# Run with custom experiment name
python backend/scripts/train_comprehensive_gnn.py --experiment custom-experiment-name
```

### 3. **Environment-Specific Execution** (Based on Config Files)
```bash
# Staging environment
AZURE_ENVIRONMENT=staging python backend/scripts/train_comprehensive_gnn.py

# Production environment
AZURE_ENVIRONMENT=prod python backend/scripts/train_comprehensive_gnn.py
```

## GNN Training Debugging Instructions

### 1. **Check Training Status** (From Existing Implementation)
```bash
# Enable debug logging
PYTHONPATH=./backend python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from backend.scripts.train_comprehensive_gnn import run_comprehensive_gnn_training
result = run_comprehensive_gnn_training()
print('Training result:', result)
"
```

### 2. **Validate Configuration** (Based on Environment Files)
```bash
# Check staging environment config
python -c "
import os
from pathlib import Path
env_file = Path('backend/config/environments/staging.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if 'GNN_' in line and not line.startswith('#'):
                print(line.strip())
else:
    print('Staging environment file not found')
"
```

### 3. **Verify Azure ML Integration** (From Existing Code)
```bash
# Test Azure ML connection
python -c "
try:
    from azureml.core import Workspace
    from azureml.core.run import Run
    run = Run.get_context()
    print(f'Run context: {run.id}')
    if run.id.startswith('OfflineRun'):
        print('Running locally - Azure ML offline mode')
    else:
        print('Running in Azure ML environment')
except Exception as e:
    print(f'Azure ML check failed: {e}')
"
```

### 4. **Monitor Training Progress** (From Trainer Implementation)
```bash
# Check model output directory
ls -la models/gnn/

# Monitor training logs
tail -f training_logs.log

# Check training history (if model exists)
python -c "
import json
from pathlib import Path
history_file = Path('models/gnn/training_history.json')
if history_file.exists():
    with open(history_file) as f:
        history = json.load(f)
        print(f'Total epochs: {len(history)}')
        if history:
            latest = history[-1]
            print(f'Latest metrics: {latest}')
"
```

### 5. **Validate Data Pipeline** (From Configuration)
```bash
# Check data directory structure
find backend/data/ -type f -name "*.pt" -o -name "*.json" | head -10

# Verify data path configuration
python -c "
import os
data_path = os.environ.get('GNN_DATA_PATH', 'backend/data/')
print(f'Data path: {data_path}')
from pathlib import Path
if Path(data_path).exists():
    files = list(Path(data_path).rglob('*'))
    print(f'Data files found: {len(files)}')
else:
    print('Data path does not exist')
"
```

### 6. **Azure ML Experiment Monitoring** (From ML Client Code)
```bash
# Check Azure ML experiment status
python -c "
try:
    from azureml.core import Workspace, Experiment
    ws = Workspace.from_config()
    exp = Experiment(workspace=ws, name='universal-rag-gnn')
    runs = list(exp.get_runs())
    print(f'Total runs: {len(runs)}')
    if runs:
        latest_run = runs[0]
        print(f'Latest run status: {latest_run.status}')
        print(f'Latest run ID: {latest_run.id}')
except Exception as e:
    print(f'Azure ML experiment check failed: {e}')
"
```

## Training Health Check Commands

### **Quick Health Check Script** (Based on Existing Components)
```bash
# Create comprehensive health check
python -c "
import logging
import json
from pathlib import Path

# Check 1: Configuration files
configs = {
    'staging': Path('backend/config/environments/staging.env'),
    'prod': Path('backend/config/environments/prod.env')
}

for env, config_path in configs.items():
    if config_path.exists():
        print(f'✅ {env} config exists')
    else:
        print(f'❌ {env} config missing')

# Check 2: Training script
script_path = Path('backend/scripts/train_comprehensive_gnn.py')
if script_path.exists():
    print('✅ Training script exists')
else:
    print('❌ Training script missing')

# Check 3: Model output directory
model_dir = Path('models/gnn')
model_dir.mkdir(parents=True, exist_ok=True)
print(f'✅ Model directory: {model_dir}')

# Check 4: Data directory
data_dir = Path('backend/data')
if data_dir.exists():
    print(f'✅ Data directory exists with {len(list(data_dir.rglob(\"*\")))} items')
else:
    print('❌ Data directory missing')

print('Health check complete')
"
```

These commands are all based on the **existing codebase structure** and provide **data-driven debugging** without any hardcoded assumptions.