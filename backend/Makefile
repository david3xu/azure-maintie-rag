# Azure Universal RAG Backend - Clean Azure Service Operations
# Session replacement pattern with Azure service integration

SHELL := /bin/bash
.DEFAULT_GOAL := help
BACKEND_SESSION := $(shell date +"%Y%m%d_%H%M%S")

# Azure Service Configuration
AZURE_ENVIRONMENT := $(or $(AZURE_ENVIRONMENT), dev)

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

.PHONY: help setup run azure-health-check data-upload knowledge-extract data-prep-full data-prep-enterprise query-demo-enterprise backend-summary clean

help: ## Azure Universal RAG Backend Operations (Two-Step Intelligent Processing)
	@echo "🔧 Azure Universal RAG Backend - Intelligent Document Processing"
	@echo "================================================================"
	@echo ""
	@echo "🚀 Backend Operations:"
	@echo "  make setup                  - Setup with Azure service validation"
	@echo "  make run                    - Start FastAPI with session tracking"
	@echo ""
	@echo "🧠 Two-Step Intelligent Data Processing:"
	@echo "  make data-upload           - STEP 1: Upload docs & create intelligent chunks"
	@echo "  make knowledge-extract     - STEP 2: Extract entities & relations from chunks"
	@echo "  make data-prep-full        - Complete workflow (STEP 1 + STEP 2)"
	@echo ""
	@echo "🎯 Azure Prompt Flow (Centralized Prompts):"
	@echo "  make prompt-flow-extract   - Extract knowledge using centralized prompts"
	@echo "  make prompt-flow-setup     - Setup Prompt Flow environment"
	@echo "  make prompt-templates      - View centralized prompt templates"
	@echo ""
	@echo "☁️  Azure Service Operations:"
	@echo "  make azure-health-check     - Validate Azure service integration"
	@echo "  make query-demo-enterprise  - Query processing with Azure tracking"
	@echo ""
	@echo "📊 Session Management:"
	@echo "  make backend-summary        - Display current backend session"
	@echo "  make clean                  - Clean session with log replacement"
	@echo ""
	@echo "📖 Legacy Commands:"
	@echo "  make data-prep-enterprise   - DEPRECATED: Use data-prep-full instead"
	@echo ""
	@echo "💡 Recommended Workflow:"
	@echo "  1. make data-upload         (Upload & chunk documents intelligently)"
	@echo "  2. make knowledge-extract   (Extract knowledge graph from chunks)"
	@echo "  OR: make data-prep-full     (Run both steps automatically)"
	@echo "  OR: make prompt-flow-extract (Use centralized prompt management)"
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

data-upload: ## STEP 1: Upload documents and create intelligent chunks
	@$(call start_backend_session)
	@echo "🔄 STEP 1: Azure Data Upload & Chunking - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,data_upload_and_chunking,scripts/data_upload_workflow.py,general)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 STEP 1: Data upload and chunking completed"
	@cat $(SESSION_SUMMARY)

knowledge-extract: ## STEP 2: Extract entities and relations from chunks
	@$(call start_backend_session)
	@echo "🔄 STEP 2: Azure Knowledge Extraction - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,knowledge_extraction,scripts/knowledge_extraction_workflow.py,general)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 STEP 2: Knowledge extraction completed"
	@cat $(SESSION_SUMMARY)

data-prep-full: ## Complete two-step data preparation (upload + extraction)
	@echo "🚀 Starting complete Azure RAG data preparation..."
	@$(MAKE) data-upload
	@echo ""
	@echo "✅ STEP 1 complete, starting STEP 2..."
	@echo ""
	@$(MAKE) knowledge-extract
	@echo ""
	@echo "🎉 Complete data preparation finished!"

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