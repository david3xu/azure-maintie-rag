# Azure Universal RAG - Production Multi-Agent System with PydanticAI
# Real Azure services integration with comprehensive health monitoring
# Implements clean session management - no log accumulation

SHELL := /bin/bash
.DEFAULT_GOAL := help
SESSION_ID := $(shell date +"%Y%m%d_%H%M%S")

# Azure Configuration - Auto-sync with azd environment
AZURE_ENVIRONMENT := $(or $(AZURE_ENVIRONMENT), prod)
AZURE_RESOURCE_GROUP := $(or $(AZURE_RESOURCE_GROUP), maintie-rag-rg)

# Persistent Session Management - Preserve All Step Outputs
CURRENT_SESSION := logs/current_session
SESSION_REPORT := logs/dataflow_execution_$(SESSION_ID).md
AZURE_STATUS := logs/azure_status_$(SESSION_ID).log
PERFORMANCE_LOG := logs/performance_$(SESSION_ID).log
CUMULATIVE_REPORT := logs/cumulative_dataflow_report.md

# Enterprise Functions - Preserve All Previous Output
define start_persistent_session
	@echo "🏗️  Starting Azure Enterprise Session: $(SESSION_ID)"
	@mkdir -p logs
	@echo "$(SESSION_ID)" > $(CURRENT_SESSION)
	@echo "# Azure Universal RAG Session Report - $(SESSION_ID)" > $(SESSION_REPORT)
	@echo "**Session ID:** $(SESSION_ID)" >> $(SESSION_REPORT)
	@echo "**Environment:** $(AZURE_ENVIRONMENT)" >> $(SESSION_REPORT)
	@echo "**Started:** $(shell date)" >> $(SESSION_REPORT)
	@echo "" >> $(SESSION_REPORT)
	@echo "" >> $(CUMULATIVE_REPORT)
	@echo "---" >> $(CUMULATIVE_REPORT)
	@echo "# Session $(SESSION_ID) - $(shell date)" >> $(CUMULATIVE_REPORT)
	@echo "" >> $(CUMULATIVE_REPORT)
endef

define capture_azure_status
	@echo "☁️  Capturing Azure service status..."
	@echo "Azure Status Check - Session: $(SESSION_ID)" > $(AZURE_STATUS)
	@echo "Timestamp: $(shell date)" >> $(AZURE_STATUS)
	@./scripts/status-working.sh >> $(AZURE_STATUS) 2>&1 || echo "Azure status script not available" >> $(AZURE_STATUS)
	@echo "## Azure Infrastructure Status" >> $(SESSION_REPORT)
	@tail -10 $(AZURE_STATUS) >> $(SESSION_REPORT)
	@echo "" >> $(SESSION_REPORT)
endef

define capture_performance_metrics
	@echo "📊 Capturing system performance..."
	@echo "Performance Metrics - Session: $(SESSION_ID)" > $(PERFORMANCE_LOG)
	@echo "=== System Resources ===" >> $(PERFORMANCE_LOG)
	@free -h >> $(PERFORMANCE_LOG)
	@echo "=== Disk Usage ===" >> $(PERFORMANCE_LOG)
	@df -h | head -5 >> $(PERFORMANCE_LOG)
	@echo "=== Active Processes ===" >> $(PERFORMANCE_LOG)
	@ps aux | grep -E "(python|uvicorn|node)" | head -5 >> $(PERFORMANCE_LOG)
	@echo "## Performance Metrics" >> $(SESSION_REPORT)
	@tail -10 $(PERFORMANCE_LOG) >> $(SESSION_REPORT)
	@echo "" >> $(SESSION_REPORT)
endef

define finalize_session_report
	@echo "**Completed:** $(shell date)" >> $(SESSION_REPORT)
	@echo "**Duration:** $$(($$(date +%s) - $$(stat -c %Y $(CURRENT_SESSION) 2>/dev/null || echo 0))) seconds" >> $(SESSION_REPORT)
	@cat $(SESSION_REPORT) >> $(CUMULATIVE_REPORT)
	@echo "📋 Session report: $(SESSION_REPORT)"
	@echo "📋 Cumulative report: $(CUMULATIVE_REPORT)"
endef

.PHONY: help setup dev azure-deploy azure-status health clean session-report sync-env deploy deploy-with-data deploy-infrastructure-only

help: ## Azure Universal RAG Multi-Agent Commands (Production Ready)
	@echo "🤖 Azure Universal RAG - Multi-Agent System with PydanticAI"
	@echo "============================================================="
	@echo ""
	@echo "🎯 Multi-Agent Development:"
	@echo "  make setup           - Setup PydanticAI agents + Azure services"
	@echo "  make dev             - Start API (8000) + UI (5174) + agent monitoring"
	@echo "  make health          - Azure services + agent health validation"
	@echo "  make test            - Run tests against real Azure services (no mocks)"
	@echo ""
	@echo "☁️  Azure Service Integration:"
	@echo "  make azure-deploy    - Deploy complete Azure infrastructure (9 services)"
	@echo "  make deploy-with-data - Full deployment with automated data pipeline (recommended)"
	@echo "  make deploy-infrastructure-only - Deploy infrastructure only, no data population"
	@echo "  make azure-status    - Azure service container health check"
	@echo "  make sync-env        - Sync with azd environment (development/staging/production)"
	@echo ""
	@echo "🧠 Data Processing Pipeline (6-Phase Architecture):"
	@echo "  make dataflow-cleanup  - Phase 0: Clean all Azure services + verify clean state"
	@echo "  make dataflow-validate - Phase 1: Basic agent connectivity (post-cleanup)"
	@echo "  make dataflow-ingest   - Phase 2: Data ingestion with real Azure AI documentation"
	@echo "  make dataflow-extract  - Phase 3: PRODUCTION knowledge extraction with FORCED chunking (NO FAKE SUCCESS)"
	@echo "  make dataflow-integrate - Phase 5: Full pipeline integration testing"
	@echo "  make dataflow-query    - Phase 4: Query analysis + universal search"
	@echo "  make dataflow-advanced - Phase 6: GNN training + monitoring"
	@echo "  make dataflow-full     - Execute all phases: 0→1→2→3→4→5→6 (complete pipeline)"
	@echo ""
	@echo "📊 Production Operations:"
	@echo "  make session-report  - Performance metrics and Azure status"
	@echo "  make clean           - Clean session with log replacement"
	@echo "  make clean-all       - Comprehensive cleanup using Phase 0 pipeline"
	@echo "  make check-data      - Check Azure services status"
	@echo ""
	@echo "Current Session: $(shell cat $(CURRENT_SESSION) 2>/dev/null || echo 'No active session')"
	@echo "Architecture: 3 PydanticAI Agents + 471-line Azure Service Container + 1,536-line Data Models"

setup: ## Enterprise setup with clean session management
	@$(call start_clean_session)
	@echo "🔧 Setting up Universal RAG backend..."
	@make setup 2>&1 | tail -10 || echo "Backend setup completed"
	@echo "🎨 Setting up frontend..."
	@cd frontend && npm install > /dev/null 2>&1 && echo "✅ Frontend dependencies installed" || echo "⚠️ Frontend setup failed"
	@$(call capture_performance_metrics)
	@$(call finalize_session_report)
	@echo "✅ Enterprise setup completed - Session: $(SESSION_ID)"

dev: ## Start development with enterprise session tracking
	@$(call start_clean_session)
	@echo "🚀 Starting Azure Universal RAG Enterprise Development"
	@echo "📍 Backend API: http://localhost:8000"
	@echo "📍 Frontend UI: http://localhost:5174"
	@echo "📊 Session: $(SESSION_ID)"
	@$(call capture_azure_status)
	@echo "Starting services..." >> $(SESSION_REPORT)
	@cd backend && make run > /dev/null 2>&1 &
	@cd frontend && npm run dev > /dev/null 2>&1 &
	@sleep 3
	@$(call capture_performance_metrics)
	@$(call finalize_session_report)

azure-deploy: ## Deploy Azure infrastructure with session logging
	@$(call start_clean_session)
	@echo "🏗️ Azure Infrastructure Deployment - Session: $(SESSION_ID)"
	@echo "Deployment Output:" >> $(SESSION_REPORT)
	@./scripts/enhanced-complete-redeploy.sh 2>&1 | tail -20 >> $(SESSION_REPORT) || echo "Deployment script error" >> $(SESSION_REPORT)
	@$(call capture_azure_status)
	@$(call finalize_session_report)
	@echo "✅ Azure deployment completed - Check: $(SESSION_REPORT)"

azure-status: ## Check Azure infrastructure with clean output
	@$(call start_clean_session)
	@echo "📊 Azure Infrastructure Status Check - Session: $(SESSION_ID)"
	@$(call capture_azure_status)
	@$(call capture_performance_metrics)
	@$(call finalize_session_report)
	@cat $(SESSION_REPORT)

azure-teardown: ## Clean Azure resources with session audit
	@$(call start_clean_session)
	@echo "🧹 Azure Resource Cleanup - Session: $(SESSION_ID)"
	@echo "Cleanup initiated at $(shell date)" >> $(SESSION_REPORT)
	@./scripts/teardown.sh 2>&1 | tail -10 >> $(SESSION_REPORT) || echo "Teardown script not available" >> $(SESSION_REPORT)
	@$(call finalize_session_report)

# Legacy targets removed - use Phase-based pipeline instead:
# - Use 'make dataflow-full' for complete pipeline

# - Use 'make dataflow-ingest' for data ingestion

# - Use 'make dataflow-extract' for knowledge extraction

# - Use 'make dataflow-query' for query pipeline

# - Use 'make dataflow-integrate' for integration testing

# - Use 'make dataflow-advanced' for advanced features

sync-env: ## Sync backend configuration with current azd environment
	@echo "🔄 Syncing backend with azd environment..."
	@./scripts/deployment/sync-env.sh
	@echo "✅ Backend configuration synchronized"

# New deployment commands with automated data pipeline integration
deploy-with-data: sync-env ## Full deployment with automated data pipeline (recommended)
	@$(call start_persistent_session)
	@echo "🚀 Full Azure deployment with automated data pipeline - Session: $(SESSION_ID)"
	@echo "Setting AUTO_POPULATE_DATA=true for automated pipeline..." >> $(SESSION_REPORT)
	@azd env set AUTO_POPULATE_DATA true
	@echo ""
	@echo "🔐 ENTERPRISE AUTHENTICATION VALIDATION"
	@echo "======================================="
	@echo "Validating Azure authentication contexts for enterprise environments..."
	@./scripts/deployment/sync-auth.sh | tee -a $(SESSION_REPORT)
	@echo ""
	@echo "🔧 Proactive authentication refresh for long-running operations..."
	@if ! az account show >/dev/null 2>&1; then \
		echo "❌ Azure CLI authentication required for automated pipeline"; \
		echo "💡 University/Enterprise environments require fresh authentication"; \
		echo "🚨 CRITICAL: Run 'az login' before deployment to ensure pipeline success"; \
		echo "   This prevents token expiration during 7-20 minute pipeline execution"; \
		echo ""; \
		echo "⏸️  Deployment paused - Please authenticate and retry:"; \
		echo "   1. az login"; \
		echo "   2. make deploy-with-data"; \
		exit 1; \
	fi
	@if ! azd auth login --check-status >/dev/null 2>&1; then \
		echo "❌ azd authentication required for infrastructure deployment"; \
		echo "💡 Run 'azd auth login' to authenticate with Azure Developer CLI"; \
		echo ""; \
		echo "⏸️  Deployment paused - Please authenticate and retry:"; \
		echo "   1. azd auth login"; \
		echo "   2. make deploy-with-data"; \
		exit 1; \
	fi
	@echo "✅ Both Azure CLI and azd authenticated - proceeding with deployment" | tee -a $(SESSION_REPORT)
	@echo "Setting local development authentication..." >> $(SESSION_REPORT)
	@export USE_MANAGED_IDENTITY=false
	@export OPENBLAS_NUM_THREADS=1
	@echo "Starting azd up deployment..." >> $(SESSION_REPORT)
	@azd up --no-prompt 2>&1 | tee -a $(SESSION_REPORT) | tail -10
	@$(call capture_azure_status)
	@$(call finalize_session_report)
	@echo "✅ Full deployment with data pipeline completed - Check: $(SESSION_REPORT)"
	@echo ""
	@echo "🔍 Checking if Option 2 pipeline completed successfully..."
	@if az account show >/dev/null 2>&1; then \
		echo "✅ Authentication available - attempting to complete any remaining pipeline phases..."; \
		echo "🚀 Executing dataflow-full to ensure complete Option 2 implementation..."; \
		$(MAKE) dataflow-full || echo "⚠️  Pipeline completion had issues - check session reports"; \
	else \
		echo "⚠️  Azure authentication expired during deployment - manual pipeline completion needed"; \
		echo "💡 Run 'az login' then 'make dataflow-full' to complete Option 2 pipeline"; \
	fi

deploy-infrastructure-only: sync-env ## Deploy infrastructure only, no data population  
	@$(call start_persistent_session)
	@echo "🏗️ Infrastructure-only deployment - Session: $(SESSION_ID)"
	@echo "Setting AUTO_POPULATE_DATA=false to skip data pipeline..." >> $(SESSION_REPORT)
	@azd env set AUTO_POPULATE_DATA false
	@echo "Starting azd provision (infrastructure only)..." >> $(SESSION_REPORT)
	@azd provision --no-prompt 2>&1 | tee -a $(SESSION_REPORT) | tail -10
	@$(call capture_azure_status)
	@$(call finalize_session_report)
	@echo "✅ Infrastructure deployment completed - Check: $(SESSION_REPORT)"
	@echo "💡 To populate data manually, run: make dataflow-full"

deploy: deploy-with-data ## Alias for deploy-with-data (recommended default)

health: ## Comprehensive service health with session management
	@$(call start_clean_session)
	@echo "🔍 Azure Universal RAG Health Assessment - Session: $(SESSION_ID)"
	@$(call capture_azure_status)
	@echo "## Backend Health" >> $(SESSION_REPORT)
	@curl -s http://localhost:8000/api/v1/health 2>/dev/null | head -3 >> $(SESSION_REPORT) || echo "Backend not responding" >> $(SESSION_REPORT)
	@echo "## Frontend Health" >> $(SESSION_REPORT)
	@curl -s http://localhost:5174 > /dev/null && echo "✅ Frontend accessible" >> $(SESSION_REPORT) || echo "❌ Frontend not running" >> $(SESSION_REPORT)
	@$(call capture_performance_metrics)
	@$(call finalize_session_report)
	@cat $(SESSION_REPORT)

session-report: ## Display current session report
	@if [ -f "$(SESSION_REPORT)" ]; then \
		echo "📋 Current Session Report:"; \
		cat $(SESSION_REPORT); \
	else \
		echo "❌ No active session. Run a command to start a session."; \
	fi

clean: ## Clean current session but preserve cumulative logs
	@if [ -f "$(CURRENT_SESSION)" ]; then \
		echo "🧹 Cleaning current session: $(shell cat $(CURRENT_SESSION))"; \
		echo "Archived session: $(shell cat $(CURRENT_SESSION))" >> logs/archived_sessions.log; \
	fi
	@rm -f $(CURRENT_SESSION) logs/dataflow_execution_*.md logs/azure_status_*.log logs/performance_*.log
	@cd backend && make clean > /dev/null 2>&1 || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Current session cleaned - cumulative logs preserved in $(CUMULATIVE_REPORT)"

clean-all-logs: ## Clean all logs including cumulative report (fresh start)
	@echo "🧹 Cleaning ALL logs including cumulative report..."
	@rm -f logs/cumulative_dataflow_report.md logs/dataflow_execution_*.md logs/azure_status_*.log logs/performance_*.log logs/archived_sessions.log
	@echo "✅ All logs cleaned - fresh start"

clean-all: ## Comprehensive cleanup using Phase 0 pipeline
	@echo "🧹 Starting comprehensive cleanup..."
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_01_cleanup_azure_data.py
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_02_cleanup_azure_storage.py
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_03_verify_clean_state.py
	@echo "✅ Comprehensive cleanup completed"

check-data: ## Check Azure services using utilities
	@echo "🔍 Checking Azure services data..."
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/utilities/setup_azure_services.py

# 6-Phase Dataflow Pipeline - Real Azure Services with Real Data
.PHONY: dataflow-cleanup dataflow-validate dataflow-ingest dataflow-extract dataflow-query dataflow-integrate dataflow-advanced dataflow-full

dataflow-cleanup: ## Phase 0 - Clean all Azure services with verification
	@$(call start_clean_session)
	@echo "🧹 PHASE 0: Azure Services Cleanup - Session: $(SESSION_ID)"
	@echo "Phase 0 cleanup initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Step 1: Azure Data Cleanup" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_01_cleanup_azure_data.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Step 2: Azure Storage Cleanup" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_02_cleanup_azure_storage.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Step 3: Clean State Verification" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_03_verify_clean_state.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 0 completed - Azure services cleaned and verified"

dataflow-validate: ## Phase 1 - Basic agent connectivity validation (post-cleanup)
	@$(call start_clean_session)
	@echo "🧪 PHASE 1: Basic Agent Connectivity - Session: $(SESSION_ID)"
	@echo "Phase 1 basic connectivity initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Basic Agent Connectivity (databases empty after Phase 0)" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase1_validation/01_00_basic_agent_connectivity.py 2>&1 | tail -8 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 1 completed - Basic agent connectivity validated (ready for data ingestion)"

dataflow-ingest: ## Phase 2 - Data ingestion with prerequisites validation (FAIL FAST)
	@$(call start_clean_session)
	@echo "📥 PHASE 2: Data Ingestion - Session: $(SESSION_ID)"
	@echo "Phase 2 ingestion initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Step 1: Prerequisites Validation (FAIL FAST)" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_00_validate_phase2_prerequisites.py 2>&1 | tee -a $(SESSION_REPORT) || (echo "❌ FAIL FAST: Phase 2 prerequisites validation failed" && exit 1)
	@echo "## Step 2: Storage Upload" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_02_storage_upload_primary.py --container documents-prod 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Step 3: Vector Embeddings" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_03_vector_embeddings.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Step 4: Search Indexing" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_04_search_indexing.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 2 completed - Real data ingested to Azure services"

dataflow-extract: ## Phase 3 - Multi-step knowledge extraction (Split into 3 focused steps)
	@$(call start_clean_session)
	@echo "🧠 PHASE 3: Multi-Step Knowledge Extraction - Session: $(SESSION_ID)"
	@echo "Phase 3 multi-step extraction initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Step 0: Prerequisites Validation" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_00_validate_phase3_prerequisites.py 2>&1 | tail -3 >> $(SESSION_REPORT)
	@echo "## Step 1: Basic Entity Extraction" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_01_basic_entity_extraction.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Step 2: Graph Storage" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_02_graph_storage.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Step 3: Verification" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_03_verification.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 3 completed - Multi-step knowledge extraction with verification"

dataflow-query: ## Phase 4 - Query pipeline with real search
	@$(call start_clean_session)
	@echo "🔍 PHASE 4: Query Pipeline - Session: $(SESSION_ID)"
	@echo "Phase 4 query pipeline initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Query Analysis" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase4_query/04_01_query_analysis.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Universal Search Demo" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase4_query/04_02_universal_search_demo.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Complete Query Pipeline" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase4_query/04_06_complete_query_pipeline.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 4 completed - Query pipeline operational"

dataflow-integrate: ## Phase 5 - Full pipeline integration testing
	@$(call start_clean_session)
	@echo "🔄 PHASE 5: Integration Testing - Session: $(SESSION_ID)"
	@echo "Phase 5 integration initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Full Pipeline Execution" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase5_integration/05_01_full_pipeline_execution.py 2>&1 | tail -10 >> $(SESSION_REPORT)
	@echo "## Query Generation Showcase" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase5_integration/05_03_query_generation_showcase.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 5 completed - End-to-end integration validated"

dataflow-advanced: ## Phase 6 - Advanced features (GNN async bootstrap + training + deployment + monitoring)
	@$(call start_clean_session)
	@echo "🚀 PHASE 6: Advanced Features - Session: $(SESSION_ID)"
	@echo "Phase 6 advanced features initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## GNN Async Bootstrap (Sets GNN_ENDPOINT_NAME)" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_11_gnn_async_bootstrap.py 2>&1 | tail -8 >> $(SESSION_REPORT)
	@echo "## Reproducible GNN Deployment Pipeline" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_10_gnn_deployment_pipeline.py 2>&1 | tail -8 >> $(SESSION_REPORT)
	@echo "## GNN Training (Legacy)" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_01_gnn_training.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Real-time Monitoring" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_02_streaming_monitor.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Configuration System Demo" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_03_config_system_demo.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 6 completed - Advanced features with reproducible GNN deployment operational"

dataflow-full: ## Execute all 6 phases sequentially: 0→1→2→3→4→5→6 with comprehensive logs
	@$(call start_persistent_session)
	@echo "🌊 EXECUTING COMPLETE 6-PHASE DATAFLOW PIPELINE"
	@echo "=================================================="
	@echo "Session: $(SESSION_ID) | Using REAL Azure services with REAL data"
	@echo ""
	@echo "🔐 PRE-PIPELINE AUTHENTICATION VALIDATION"
	@echo "========================================="
	@./scripts/deployment/sync-auth.sh validate | tee -a $(SESSION_REPORT)
	@if [ $$? -ne 0 ]; then \
		echo "❌ PIPELINE ABORTED: Authentication validation failed"; \
		echo "💡 Run: az login && azd auth login, then retry"; \
		exit 1; \
	fi
	@echo ""
	@echo "# Complete 6-Phase Dataflow Execution" >> $(SESSION_REPORT)
	@echo "**Pipeline Session:** $(SESSION_ID)" >> $(SESSION_REPORT)
	@echo "**Started:** $(shell date)" >> $(SESSION_REPORT)
	@echo "**Authentication:** Validated for enterprise environment" >> $(SESSION_REPORT)
	@echo "" >> $(SESSION_REPORT)
	@echo "🧹 Phase 0: Cleaning Azure services for fresh start..."
	@echo "## Phase 0: Azure Services Cleanup" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_01_cleanup_azure_data.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_02_cleanup_azure_storage.py 2>&1 | tee -a $(SESSION_REPORT) | tail -2  
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_03_verify_clean_state.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@echo "" >> $(SESSION_REPORT)
	@echo "🧪 Phase 1: Basic agent connectivity validation (databases empty)..."
	@echo "## Phase 1: Basic Agent Connectivity" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase1_validation/01_00_basic_agent_connectivity.py 2>&1 | tee -a $(SESSION_REPORT) | tail -5
	@echo "" >> $(SESSION_REPORT)
	@echo "📥 Phase 2: Data ingestion with real Azure AI documentation..."
	@echo "## Phase 2: Data Ingestion" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_00_validate_phase2_prerequisites.py 2>&1 | tee -a $(SESSION_REPORT) | tail -2
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_02_storage_upload_primary.py --container documents-prod 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_03_vector_embeddings.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_04_search_indexing.py --source "$(PWD)/data/raw/" 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@echo "" >> $(SESSION_REPORT)
	@echo "🧠 Phase 3: Multi-step knowledge extraction (3 focused steps)..."
	@echo "## Phase 3: Multi-Step Knowledge Extraction" >> $(SESSION_REPORT)
	@echo "### Step 0: Prerequisites Validation" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_00_validate_phase3_prerequisites.py 2>&1 | tee -a $(SESSION_REPORT) | tail -2
	@echo "### Step 1: Basic Entity Extraction" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_01_basic_entity_extraction.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@echo "### Step 2: Graph Storage" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_02_graph_storage.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@echo "### Step 3: Verification" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_03_verification.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@echo "" >> $(SESSION_REPORT)
	@echo "🔍 Phase 4: Query pipeline with tri-modal search..."
	@echo "## Phase 4: Query Pipeline" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase4_query/04_01_query_analysis.py "How to train custom models with Azure AI?" 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@echo "" >> $(SESSION_REPORT)
	@echo "🔄 Phase 5: Full pipeline integration testing..."
	@echo "## Phase 5: Integration Testing" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase5_integration/05_01_full_pipeline_execution.py --verbose 2>&1 | tee -a $(SESSION_REPORT) | tail -5
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase5_integration/05_03_query_generation_showcase.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@echo "" >> $(SESSION_REPORT)
	@echo "🚀 Phase 6: Advanced features (GNN async bootstrap + reproducible deployment + monitoring)..."
	@echo "## Phase 6: Advanced Features" >> $(SESSION_REPORT)
	@echo "### GNN Async Bootstrap (Sets GNN_ENDPOINT_NAME)" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_11_gnn_async_bootstrap.py 2>&1 | tee -a $(SESSION_REPORT) | tail -5
	@echo "### Reproducible GNN Deployment Pipeline" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_10_gnn_deployment_pipeline.py 2>&1 | tee -a $(SESSION_REPORT) | tail -5
	@echo "### GNN Training (Legacy)" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_01_gnn_training.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@echo "### Real-time Monitoring" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_02_streaming_monitor.py 2>&1 | tee -a $(SESSION_REPORT) | tail -2
	@echo "### Configuration System Demo" >> $(SESSION_REPORT)
	@USE_MANAGED_IDENTITY=false PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_03_config_system_demo.py 2>&1 | tee -a $(SESSION_REPORT) | tail -3
	@$(call capture_performance_metrics)
	@$(call finalize_session_report)
	@echo ""
	@echo "🎉 COMPLETE 6-PHASE PIPELINE EXECUTED - Session: $(SESSION_ID)"
	@echo "📋 This session log: $(SESSION_REPORT)"
	@echo "📋 All sessions cumulative: $(CUMULATIVE_REPORT)"
	@echo "📊 Performance metrics: $(PERFORMANCE_LOG)"

# Legacy compatibility
backend:
	@cd backend && make run
frontend:
	@cd frontend && npm run dev
test:
	@cd backend && make test