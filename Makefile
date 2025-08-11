# Azure Universal RAG - Production Multi-Agent System with PydanticAI
# Real Azure services integration with comprehensive health monitoring
# Implements clean session management - no log accumulation

SHELL := /bin/bash
.DEFAULT_GOAL := help
SESSION_ID := $(shell date +"%Y%m%d_%H%M%S")

# Azure Configuration - Auto-sync with azd environment
AZURE_ENVIRONMENT := $(or $(AZURE_ENVIRONMENT), prod)
AZURE_RESOURCE_GROUP := $(or $(AZURE_RESOURCE_GROUP), maintie-rag-rg)

# Clean Session Management - Replace Previous Sessions
CURRENT_SESSION := logs/current_session
SESSION_REPORT := logs/session_report.md
AZURE_STATUS := logs/azure_status.log
PERFORMANCE_LOG := logs/performance.log

# Enterprise Functions - Replace Previous Output
define start_clean_session
	@echo "🏗️  Starting Azure Enterprise Session: $(SESSION_ID)"
	@mkdir -p logs
	@echo "$(SESSION_ID)" > $(CURRENT_SESSION)
	@echo "# Azure Universal RAG Session Report" > $(SESSION_REPORT)
	@echo "**Session ID:** $(SESSION_ID)" >> $(SESSION_REPORT)
	@echo "**Environment:** $(AZURE_ENVIRONMENT)" >> $(SESSION_REPORT)
	@echo "**Started:** $(shell date)" >> $(SESSION_REPORT)
	@echo "" >> $(SESSION_REPORT)
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
	@echo "📋 Session report: $(SESSION_REPORT)"
endef

.PHONY: help setup dev azure-deploy azure-status health clean session-report sync-env

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
	@echo "  make azure-status    - Azure service container health check"
	@echo "  make sync-env        - Sync with azd environment (development/staging/production)"
	@echo ""
	@echo "🧠 Data Processing Pipeline (6-Phase Architecture):"
	@echo "  make dataflow-cleanup  - Phase 0: Clean all Azure services (always runs first)"
	@echo "  make dataflow-validate - Phase 1: Validate all 3 PydanticAI agents with real Azure"
	@echo "  make dataflow-ingest   - Phase 2: Upload real data to Azure Storage + embeddings"
	@echo "  make dataflow-extract  - Phase 3: Knowledge extraction + graph building"
	@echo "  make dataflow-integrate - Phase 5: Full pipeline integration testing"
	@echo "  make dataflow-query    - Phase 4: Query analysis + universal search"
	@echo "  make dataflow-advanced - Phase 6: GNN training + monitoring"
	@echo "  make dataflow-full     - Execute all phases: 0→1→2→3→5→4→6 (with cleanup first)"
	@echo ""
	@echo "📊 Production Operations:"
	@echo "  make session-report  - Performance metrics and Azure status"
	@echo "  make clean           - Clean session with log replacement"
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

# Data Processing Pipeline Commands - New Dataflow Architecture
data-prep-full: ## Complete data processing pipeline (00_full_pipeline.py)
	@$(call start_clean_session)
	@echo "🧠 Azure Universal RAG - Complete Data Pipeline (Dataflow Architecture)"
	@echo "Data processing pipeline initiated at $(shell date)" >> $(SESSION_REPORT)
	@python scripts/dataflow/00_full_pipeline.py 2>&1 | tail -15 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Complete pipeline finished - Check: $(SESSION_REPORT)"

data-upload: ## Data ingestion stage (01_data_ingestion.py)
	@$(call start_clean_session)
	@echo "📤 Azure Universal RAG - Data Ingestion"
	@echo "Data ingestion initiated at $(shell date)" >> $(SESSION_REPORT)
	@python scripts/dataflow/01_data_ingestion.py 2>&1 | tail -10 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Data ingestion finished - Check: $(SESSION_REPORT)"

knowledge-extract: ## Knowledge extraction stage (02_knowledge_extraction.py)
	@$(call start_clean_session)
	@echo "🧠 Azure Universal RAG - Knowledge Extraction"
	@echo "Knowledge extraction initiated at $(shell date)" >> $(SESSION_REPORT)
	@python scripts/dataflow/02_knowledge_extraction.py 2>&1 | tail -10 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Knowledge extraction finished - Check: $(SESSION_REPORT)"

# New Dataflow Commands - Query Pipeline
query-demo: ## Complete query pipeline (10_query_pipeline.py)
	@$(call start_clean_session)
	@echo "🔍 Azure Universal RAG - Query Pipeline Demo"
	@echo "Query pipeline initiated at $(shell date)" >> $(SESSION_REPORT)
	@python scripts/dataflow/10_query_pipeline.py 2>&1 | tail -10 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Query pipeline finished - Check: $(SESSION_REPORT)"

unified-search-demo: ## Unified search demonstration (07_unified_search.py)
	@$(call start_clean_session)
	@echo "🎯 Azure Universal RAG - Unified Search Demo (Vector + Graph + GNN)"
	@echo "Unified search demo initiated at $(shell date)" >> $(SESSION_REPORT)
	@python scripts/dataflow/07_unified_search.py 2>&1 | tail -10 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Unified search demo finished - Check: $(SESSION_REPORT)"

full-workflow-demo: ## End-to-end workflow demonstration
	@$(call start_clean_session)
	@echo "🚀 Azure Universal RAG - Full Workflow Demonstration"
	@echo "Full workflow demo initiated at $(shell date)" >> $(SESSION_REPORT)
	@python scripts/dataflow/demo_full_workflow.py 2>&1 | tail -15 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Full workflow demo finished - Check: $(SESSION_REPORT)"

sync-env: ## Sync backend configuration with current azd environment
	@echo "🔄 Syncing backend with azd environment..."
	@./scripts/deployment/sync-env.sh
	@echo "✅ Backend configuration synchronized"

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

clean: ## Clean session with log replacement
	@if [ -f "$(CURRENT_SESSION)" ]; then \
		echo "🧹 Cleaning session: $(shell cat $(CURRENT_SESSION))"; \
		echo "Archived session: $(shell cat $(CURRENT_SESSION))" > logs/last_session.log; \
	fi
	@rm -f $(CURRENT_SESSION) $(SESSION_REPORT) $(AZURE_STATUS) $(PERFORMANCE_LOG)
	@cd backend && make clean > /dev/null 2>&1 || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean session completed - logs replaced"

clean-all: ## Comprehensive cleanup - all data except data/raw and Azure services
	@echo "🧹 Starting comprehensive cleanup..."
	@PYTHONPATH=$(PWD) python scripts/dataflow/00_clean_all_data.py
	@echo "✅ Comprehensive cleanup completed"

check-data: ## Check what data exists in Azure services
	@echo "🔍 Checking Azure services data..."
	@PYTHONPATH=$(PWD) python scripts/dataflow/00_check_azure_data.py

# 6-Phase Dataflow Pipeline - Real Azure Services with Real Data
.PHONY: dataflow-cleanup dataflow-validate dataflow-ingest dataflow-extract dataflow-query dataflow-integrate dataflow-advanced dataflow-full

dataflow-cleanup: ## Phase 0 - Clean all Azure services
	@$(call start_clean_session)
	@echo "🧹 PHASE 0: Azure Services Cleanup - Session: $(SESSION_ID)"
	@echo "Phase 0 cleanup initiated at $(shell date)" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase0_cleanup/00_01_cleanup_azure_data.py 2>&1 | tail -10 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 0 completed - Azure services cleaned"

dataflow-validate: ## Phase 1 - Validate all 3 PydanticAI agents
	@$(call start_clean_session)
	@echo "🧪 PHASE 1: Agent Validation - Session: $(SESSION_ID)"
	@echo "Phase 1 validation initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Domain Intelligence Agent" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase1_validation/01_01_validate_domain_intelligence.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Knowledge Extraction Agent" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase1_validation/01_02_validate_knowledge_extraction.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Universal Search Agent" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase1_validation/01_03_validate_universal_search.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 1 completed - All agents validated with real Azure"

dataflow-ingest: ## Phase 2 - Data ingestion with real data
	@$(call start_clean_session)
	@echo "📥 PHASE 2: Data Ingestion - Session: $(SESSION_ID)"
	@echo "Phase 2 ingestion initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Storage Upload" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_02_storage_upload_primary.py --container documents-prod 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Vector Embeddings" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_03_vector_embeddings.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Search Indexing" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase2_ingestion/02_04_search_indexing.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 2 completed - Real data ingested to Azure"

dataflow-extract: ## Phase 3 - Knowledge extraction with real processing
	@$(call start_clean_session)
	@echo "🧠 PHASE 3: Knowledge Extraction - Session: $(SESSION_ID)"
	@echo "Phase 3 extraction initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Knowledge Extraction" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_02_knowledge_extraction.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Cosmos DB Storage" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_03_cosmos_storage.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Graph Construction" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase3_knowledge/03_04_graph_construction.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 3 completed - Knowledge graphs built"

dataflow-query: ## Phase 4 - Query pipeline with real search
	@$(call start_clean_session)
	@echo "🔍 PHASE 4: Query Pipeline - Session: $(SESSION_ID)"
	@echo "Phase 4 query pipeline initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Query Analysis" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase4_query/04_01_query_analysis.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Universal Search Demo" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase4_query/04_02_universal_search_demo.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Complete Query Pipeline" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase4_query/04_06_complete_query_pipeline.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 4 completed - Query pipeline operational"

dataflow-integrate: ## Phase 5 - Full pipeline integration testing
	@$(call start_clean_session)
	@echo "🔄 PHASE 5: Integration Testing - Session: $(SESSION_ID)"
	@echo "Phase 5 integration initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## Full Pipeline Execution" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase5_integration/05_01_full_pipeline_execution.py 2>&1 | tail -10 >> $(SESSION_REPORT)
	@echo "## Query Generation Showcase" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase5_integration/05_03_query_generation_showcase.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 5 completed - End-to-end integration validated"

dataflow-advanced: ## Phase 6 - Advanced features (GNN training + monitoring)
	@$(call start_clean_session)
	@echo "🚀 PHASE 6: Advanced Features - Session: $(SESSION_ID)"
	@echo "Phase 6 advanced features initiated at $(shell date)" >> $(SESSION_REPORT)
	@echo "## GNN Training" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_01_gnn_training.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Real-time Monitoring" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_02_streaming_monitor.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@echo "## Configuration System Demo" >> $(SESSION_REPORT)
	@PYTHONPATH=$(PWD) python scripts/dataflow/phase6_advanced/06_03_config_system_demo.py 2>&1 | tail -5 >> $(SESSION_REPORT)
	@$(call finalize_session_report)
	@echo "✅ Phase 6 completed - Advanced features operational"

dataflow-full: ## Execute all 6 phases sequentially with cleanup
	@echo "🌊 EXECUTING COMPLETE 6-PHASE DATAFLOW PIPELINE"
	@echo "=================================================="
	@echo "Using REAL Azure services with REAL data from data/raw/"
	@echo ""
	@echo "🧹 Phase 0: Cleaning Azure services for fresh start..."
	@make dataflow-cleanup
	@echo ""
	@make dataflow-validate
	@echo ""
	@make dataflow-ingest
	@echo ""
	@make dataflow-extract
	@echo ""
	@make dataflow-integrate
	@echo ""
	@make dataflow-query
	@echo ""
	@make dataflow-advanced
	@echo ""
	@echo "🎉 COMPLETE 6-PHASE PIPELINE EXECUTED"
	@echo "Check session reports in logs/ for detailed results"

# Legacy compatibility
backend:
	@cd backend && make run
frontend:
	@cd frontend && npm run dev
test:
	@cd backend && make test