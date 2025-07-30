# Azure Universal RAG Backend - Clean Azure Service Operations
# Session replacement pattern with Azure service integration

SHELL := /bin/bash
.DEFAULT_GOAL := help
BACKEND_SESSION := $(shell date +"%Y%m%d_%H%M%S")

# Azure Service Configuration
AZURE_ENVIRONMENT := $(or $(AZURE_ENVIRONMENT), staging)

# Clean Session Files - Replace Previous Output
CURRENT_BACKEND_SESSION := logs/backend_session.current
AZURE_HEALTH_LOG := logs/azure_health.log
WORKFLOW_LOG := logs/workflow.log
SESSION_SUMMARY := logs/backend_summary.md

# Azure Service Integration Functions
define start_backend_session
	@echo "🔧 Starting Azure RAG Backend Session: $(BACKEND_SESSION)"
	@mkdir -p logs
	@echo "$(BACKEND_SESSION)" > $(CURRENT_BACKEND_SESSION)
	@echo "# Azure Universal RAG Backend Session" > $(SESSION_SUMMARY)
	@echo "**Backend Session:** $(BACKEND_SESSION)" >> $(SESSION_SUMMARY)
	@echo "**Environment:** $(AZURE_ENVIRONMENT)" >> $(SESSION_SUMMARY)
	@echo "**Started:** $(shell date)" >> $(SESSION_SUMMARY)
	@echo "" >> $(SESSION_SUMMARY)
endef

define validate_azure_services
	@echo "🔍 Validating Azure service integration..."
	@echo "Azure Service Validation - Session: $(BACKEND_SESSION)" > $(AZURE_HEALTH_LOG)
	@echo "Timestamp: $(shell date)" >> $(AZURE_HEALTH_LOG)
	@PYTHONPATH=. python3 scripts/azure_config_validator.py >> $(AZURE_HEALTH_LOG) 2>&1 || echo "Azure validation failed" >> $(AZURE_HEALTH_LOG)
	@echo "## Azure Service Health" >> $(SESSION_SUMMARY)
	@grep -E "(Overall Status|Service Health Ratio|OPERATIONAL READINESS)" $(AZURE_HEALTH_LOG) >> $(SESSION_SUMMARY) 2>/dev/null || echo "Service health data unavailable" >> $(SESSION_SUMMARY)
	@echo "" >> $(SESSION_SUMMARY)
endef

define execute_azure_workflow
	@echo "⚙️  Executing Azure workflow: $(1)"
	@echo "Azure Workflow Execution - $(1)" > $(WORKFLOW_LOG)
	@echo "Session: $(BACKEND_SESSION) | Start: $(shell date)" >> $(WORKFLOW_LOG)
	@PYTHONPATH=. python3 $(2) $(3) >> $(WORKFLOW_LOG) 2>&1 || echo "Workflow execution failed" >> $(WORKFLOW_LOG)
	@echo "Session: $(BACKEND_SESSION) | End: $(shell date)" >> $(WORKFLOW_LOG)
	@echo "## Workflow: $(1)" >> $(SESSION_SUMMARY)
	@grep -E "(completed|Entities|Relations|Duration)" $(WORKFLOW_LOG) | tail -5 >> $(SESSION_SUMMARY) 2>/dev/null || echo "Workflow data unavailable" >> $(SESSION_SUMMARY)
	@echo "" >> $(SESSION_SUMMARY)
endef

define capture_backend_metrics
	@echo "📊 Capturing backend performance metrics..."
	@echo "=== Backend Performance ===" >> $(SESSION_SUMMARY)
	@ps aux | grep -E "(python|uvicorn)" | wc -l | xargs echo "Python processes:" >> $(SESSION_SUMMARY)
	@free -h | grep "Mem:" >> $(SESSION_SUMMARY)
	@echo "" >> $(SESSION_SUMMARY)
endef

define finalize_backend_session
	@echo "**Completed:** $(shell date)" >> $(SESSION_SUMMARY)
	@echo "**Duration:** $$(($$(date +%s) - $$(stat -c %Y $(CURRENT_BACKEND_SESSION) 2>/dev/null || echo 0))) seconds" >> $(SESSION_SUMMARY)
	@echo "📋 Backend session summary: $(SESSION_SUMMARY)"
endef

.PHONY: help setup run azure-health-check data-upload knowledge-extract data-prep-full data-prep-enterprise query-demo-enterprise backend-summary clean clean-data

help: ## Azure Universal RAG Backend - Dataflow Architecture
	@echo "🔧 Azure Universal RAG Backend - Living Dataflow Architecture"
	@echo "=============================================================="
	@echo ""
	@echo "🚀 Backend Operations:"
	@echo "  make setup                  - Setup with Azure service validation"
	@echo "  make run                    - Start FastAPI with session tracking"
	@echo ""
	@echo "🧠 Complete Pipeline Commands (New Dataflow Architecture):"
	@echo "  make data-prep-full         - Complete data pipeline (00_full_pipeline.py)"
	@echo "  make query-pipeline         - Complete query pipeline (10_query_pipeline.py)"
	@echo "  make full-workflow-demo     - End-to-end demonstration"
	@echo ""
	@echo "📊 Individual Processing Stages:"
	@echo "  make data-ingestion         - Data ingestion (01_data_ingestion.py)"
	@echo "  make knowledge-extract      - Knowledge extraction (02_knowledge_extraction.py)"
	@echo "  make vector-indexing        - Vector embeddings (01c_vector_embeddings.py)"
	@echo "  make graph-construction     - Graph construction (04_graph_construction.py)"
	@echo "  make gnn-training           - GNN training (05_gnn_training.py)"
	@echo ""
	@echo "🔍 Query Processing Stages:"
	@echo "  make query-analysis         - Query analysis (06_query_analysis.py)"
	@echo "  make unified-search         - Unified search demo (07_unified_search.py)"
	@echo ""
	@echo "☁️  Azure Service Operations:"
	@echo "  make azure-health-check     - Validate Azure service integration"
	@echo ""
	@echo "📊 Session Management:"
	@echo "  make backend-summary        - Display current backend session"
	@echo "  make clean                  - Clean session with log replacement"
	@echo "  make clean-data             - Clean processed data (preserve raw)"
	@echo ""
	@echo "💡 Recommended Workflows:"
	@echo ""
	@echo "🔄 Full Pipeline (Automated):"
	@echo "  make data-prep-full         - Complete processing pipeline"
	@echo "  make query-pipeline         - Demonstrate query capabilities"
	@echo ""
	@echo "🔬 Stage-by-Stage (Manual Control):"
	@echo "  make data-ingestion → knowledge-extract → graph-construction → gnn-training"
	@echo "  make query-analysis → unified-search"
	@echo ""
	@echo "🎯 Unified Search Demo (Crown Jewel):"
	@echo "  make unified-search         - Vector + Graph + GNN search demonstration"
	@echo ""
	@echo "Current Backend Session: $(shell cat $(CURRENT_BACKEND_SESSION) 2>/dev/null || echo 'No active session')"

setup: ## Setup with comprehensive Azure service validation
	@$(call start_backend_session)
	@echo "🔧 Azure Universal RAG Backend Setup - Session: $(BACKEND_SESSION)"
	@python3 -m venv .venv > /dev/null 2>&1 || echo ".venv already exists"
	@./.venv/bin/python -m pip install --upgrade pip > /dev/null 2>&1
	@./.venv/bin/python -m pip install -r requirements.txt > /dev/null 2>&1 && echo "✅ Dependencies installed" >> $(SESSION_SUMMARY) || echo "⚠️ Dependency installation failed" >> $(SESSION_SUMMARY)
	@mkdir -p data/{raw,processed,indices,cache,output,metrics} 2>/dev/null || true
	@$(call validate_azure_services)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "✅ Azure backend setup completed - Session: $(BACKEND_SESSION)"

run: ## Start FastAPI with Azure service monitoring
	@$(call start_backend_session)
	@echo "🚀 Starting Azure Universal RAG Backend API"
	@echo "📍 Backend API: http://localhost:8000"
	@echo "📍 API Documentation: http://localhost:8000/docs"
	@echo "📊 Backend Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@echo "FastAPI Server Status: Starting" >> $(SESSION_SUMMARY)
	@uvicorn main:app --reload --host 0.0.0.0 --port 8000

azure-health-check: ## Comprehensive Azure service validation
	@$(call start_backend_session)
	@echo "🔍 Azure Universal RAG Service Health Check - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📊 Azure service health check completed"
	@cat $(SESSION_SUMMARY)

# === New Dataflow Architecture Commands ===

# Full Pipeline Commands
data-prep-full: ## Complete data processing pipeline (00_full_pipeline.py)
	@$(call start_backend_session)
	@echo "🚀 Complete Azure RAG Data Pipeline - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,complete_data_pipeline,scripts/dataflow/00_full_pipeline.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Complete data pipeline finished"
	@cat $(SESSION_SUMMARY)

query-pipeline: ## Complete query processing pipeline (10_query_pipeline.py)
	@$(call start_backend_session)
	@echo "🔍 Complete Query Pipeline - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,query_pipeline,scripts/dataflow/10_query_pipeline.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Query pipeline completed"
	@cat $(SESSION_SUMMARY)

# Individual Stage Commands
data-ingestion: ## Data ingestion stage (01_data_ingestion.py)
	@$(call start_backend_session)
	@echo "📤 Data Ingestion - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,data_ingestion,scripts/dataflow/01_data_ingestion.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Data ingestion completed"
	@cat $(SESSION_SUMMARY)

knowledge-extract: ## Knowledge extraction stage (02_knowledge_extraction.py)
	@$(call start_backend_session)
	@echo "🧠 Knowledge Extraction - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,knowledge_extraction,scripts/dataflow/02_knowledge_extraction.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Knowledge extraction completed"
	@cat $(SESSION_SUMMARY)

vector-indexing: ## Vector embeddings and indexing (01c_vector_embeddings.py)
	@$(call start_backend_session)
	@echo "🔍 Vector Indexing - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,vector_indexing,scripts/dataflow/01c_vector_embeddings.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Vector indexing completed"
	@cat $(SESSION_SUMMARY)

graph-construction: ## Graph construction stage (04_graph_construction.py)
	@$(call start_backend_session)
	@echo "🕸️  Graph Construction - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,graph_construction,scripts/dataflow/04_graph_construction.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Graph construction completed"
	@cat $(SESSION_SUMMARY)

gnn-training: ## GNN training stage (05_gnn_training.py)
	@$(call start_backend_session)
	@echo "🤖 GNN Training - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,gnn_training,scripts/dataflow/05_gnn_training.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 GNN training completed"
	@cat $(SESSION_SUMMARY)

# Query Stage Commands
query-analysis: ## Query analysis stage (06_query_analysis.py)
	@$(call start_backend_session)
	@echo "🔍 Query Analysis - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,query_analysis,scripts/dataflow/06_query_analysis.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Query analysis completed"
	@cat $(SESSION_SUMMARY)

unified-search: ## Unified search demo (07_unified_search.py)
	@$(call start_backend_session)
	@echo "🎯 Unified Search (Vector + Graph + GNN) - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,unified_search,scripts/dataflow/07_unified_search.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Unified search completed"
	@cat $(SESSION_SUMMARY)

# Demo Commands
full-workflow-demo: ## End-to-end workflow demonstration
	@$(call start_backend_session)
	@echo "🚀 Full Workflow Demo - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,full_workflow_demo,scripts/dataflow/demo_full_workflow.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Full workflow demo completed"
	@cat $(SESSION_SUMMARY)

# Legacy compatibility aliases
data-upload: data-ingestion

data-prep-enterprise: ## Enhanced data preparation with real-time progress tracking
	@echo "🚀 Starting enhanced Azure RAG data preparation with real-time progress..."
	@$(call start_backend_session)
	@echo "🔄 STEP 1: Azure Data Upload & Chunking - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,data_upload_and_chunking,scripts/data_preparation_workflow.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 STEP 1: Data upload and chunking completed"
	@cat $(SESSION_SUMMARY)

query-demo-enterprise: ## Query processing with Azure performance tracking
	@$(call start_backend_session)
	@echo "🔄 Azure Enterprise Query Processing - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,query_processing,scripts/query_processing_workflow.py,)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Enterprise query processing completed"
	@cat $(SESSION_SUMMARY)

data-state:
	PYTHONPATH=. python scripts/azure_data_state.py

azure-clean-data: ## Enterprise Azure data cleanup preserving infrastructure
	@$(call start_backend_session)
	@echo "🧹 Azure Data Cleanup Service Orchestration - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,data_cleanup,scripts/azure_data_cleanup_workflow.py,general)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Azure data cleanup completed"
	@cat $(SESSION_SUMMARY)

data-clean: ## Complete data lifecycle reset
	@echo "🔄 Enterprise data lifecycle reset..."
	make clean
	make azure-clean-data

full-dataset-extract: ## Full dataset knowledge extraction with real-time progress
	@$(call start_backend_session)
	@echo "🌐 Full Dataset Knowledge Extraction - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,full_dataset_extraction,scripts/full_dataset_extraction.py,--batch-size 20 --no-details)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Full dataset extraction completed"
	@cat $(SESSION_SUMMARY)

backend-summary: ## Display current backend session summary
	@if [ -f "$(SESSION_SUMMARY)" ]; then \
		echo "📋 Current Backend Session Summary:"; \
		cat $(SESSION_SUMMARY); \
	else \
		echo "❌ No active backend session. Run a backend command to start a session."; \
	fi

clean: ## Clean backend session with log replacement
	@if [ -f "$(CURRENT_BACKEND_SESSION)" ]; then \
		echo "🧹 Cleaning backend session: $(shell cat $(CURRENT_BACKEND_SESSION))"; \
		echo "Archived backend session: $(shell cat $(CURRENT_BACKEND_SESSION))" > logs/last_backend_session.log; \
	fi
	@rm -f $(CURRENT_BACKEND_SESSION) $(AZURE_HEALTH_LOG) $(WORKFLOW_LOG) $(SESSION_SUMMARY)
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find data/processed -type f -delete 2>/dev/null || true
	@find data/cache -type f -delete 2>/dev/null || true
	@echo "✅ Backend session cleaned - logs replaced"

# Azure Prompt Flow Commands (Centralized Prompt Management)
prompt-flow-extract: ## Extract knowledge using Azure Prompt Flow with centralized prompts
	@$(call start_backend_session)
	@echo "🎯 Azure Prompt Flow Knowledge Extraction - Session: $(BACKEND_SESSION)"
	@echo "Using centralized prompt templates - NO predetermined knowledge!"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,prompt_flow_extraction,scripts/prompt_flow_knowledge_extraction.py,general)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Prompt Flow extraction completed"
	@cat $(SESSION_SUMMARY)

prompt-flow-setup: ## Setup Azure Prompt Flow environment and dependencies
	@$(call start_backend_session)
	@echo "🔧 Setting up Azure Prompt Flow environment - Session: $(BACKEND_SESSION)"
	@pip install promptflow promptflow-tools promptflow-azure || echo "⚠️ Prompt Flow installation may require additional setup"
	@echo "📁 Prompt Flow directory: prompt_flows/universal_knowledge_extraction/"
	@echo "📋 Centralized templates:"
	@ls -la prompt_flows/universal_knowledge_extraction/*.jinja2 2>/dev/null || echo "   No templates found"
	@$(call finalize_backend_session)
	@echo "✅ Prompt Flow setup completed"

prompt-templates: ## View and manage centralized prompt templates
	@echo "📋 Azure Prompt Flow - Centralized Universal Prompt Templates"
	@echo "=============================================================="
	@echo ""
	@echo "🎯 Universal Extraction Principles:"
	@echo "   ✅ NO predetermined entity types"
	@echo "   ✅ NO hardcoded domain knowledge"
	@echo "   ✅ Domain-agnostic prompt templates"
	@echo ""
	@echo "📁 Template Location: prompt_flows/universal_knowledge_extraction/"
	@echo ""
	@echo "📄 Available Templates:"
	@find prompt_flows/universal_knowledge_extraction/ -name "*.jinja2" -exec basename {} \; 2>/dev/null | sed 's/^/   ✅ /' || echo "   ❌ No templates found"
	@echo ""
	@echo "🔧 Template Management:"
	@echo "   - Edit templates directly in prompt_flows/universal_knowledge_extraction/"
	@echo "   - Templates use Jinja2 syntax for dynamic content"
	@echo "   - Changes take effect immediately (no restart required)"
	@echo ""
	@echo "💡 Usage: make prompt-flow-extract (uses centralized templates)"

# Legacy compatibility with session enhancement
data-prep: data-prep-enterprise
query-demo: query-demo-enterprise
test: azure-health-check

# === Data Management ===

clean-data: ## Clean processed data (preserve raw and back_data)
	@echo "🧹 Cleaning processed data..."
	@for dir in demo demo_outputs extraction_comparisons extraction_outputs extraction_progress gnn_models gnn_training indices kg_operations loading_results; do \
		if [ -d "data/$$dir" ]; then \
			rm -rf "data/$$dir"; \
		fi; \
	done
	@if [ -d "data/outputs" ]; then rm -rf data/outputs/*; fi
	@if [ -d "logs" ]; then rm -rf logs/*; fi
	@echo "✅ Processed data cleaned (raw and back_data preserved)"