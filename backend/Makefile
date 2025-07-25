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

.PHONY: help setup run azure-health-check data-prep-enterprise query-demo-enterprise clean

help: ## Azure Universal RAG Backend Operations (Clean Session Management)
	@echo "🔧 Azure Universal RAG Backend - Clean Azure Service Operations"
	@echo "=============================================================="
	@echo ""
	@echo "🚀 Backend Operations:"
	@echo "  make setup                  - Setup with Azure service validation"
	@echo "  make run                    - Start FastAPI with session tracking"
	@echo ""
	@echo "☁️  Azure Service Operations:"
	@echo "  make azure-health-check     - Validate Azure service integration"
	@echo "  make data-prep-enterprise   - Data preparation with Azure services"
	@echo "  make query-demo-enterprise  - Query processing with Azure tracking"
	@echo ""
	@echo "📊 Session Management:"
	@echo "  make backend-summary        - Display current backend session"
	@echo "  make clean                  - Clean session with log replacement"
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

data-prep-enterprise: ## Enterprise data preparation with Azure service integration
	@$(call start_backend_session)
	@echo "🔄 Azure Enterprise Data Preparation - Session: $(BACKEND_SESSION)"
	@$(call validate_azure_services)
	@$(call execute_azure_workflow,data_preparation,scripts/azure_knowledge_graph_service.py,general)
	@$(call capture_backend_metrics)
	@$(call finalize_backend_session)
	@echo "📋 Enterprise data preparation completed"
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

# Legacy compatibility with session enhancement
data-prep: data-prep-enterprise
query-demo: query-demo-enterprise
test: azure-health-check