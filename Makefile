# Azure Universal RAG - Production Makefile
# Uses REAL Azure services, REAL data from data/raw/, NO fake patterns

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Essential Configuration
PYTHONPATH := $(PWD)
USE_MANAGED_IDENTITY := false
OPENBLAS_NUM_THREADS := 1
export PYTHONPATH USE_MANAGED_IDENTITY OPENBLAS_NUM_THREADS

help: ## Show available commands
	@echo "🚀 Azure Universal RAG - Production Commands"
	@echo "============================================"
	@echo ""
	@echo "📋 Quick Start:"
	@echo "  make setup              - Install all dependencies"
	@echo "  make dev                - Start API + Frontend servers"
	@echo "  make health             - System health check"
	@echo ""
	@echo "🧪 Testing & Validation:"
	@echo "  make test               - Run tests with REAL Azure services"
	@echo "  make dataflow-validate  - Validate all 3 agents connectivity"
	@echo "  make dataflow-full      - Complete 6-phase pipeline (cleanup→validate→ingest→extract→query→integrate→advanced)"
	@echo ""
	@echo "☁️ Deployment:"
	@echo "  make deploy             - Deploy infrastructure only (fast, 2-3 min)"
	@echo "  make deploy-with-data   - Deploy + full data pipeline (10-15 min)"
	@echo ""
	@echo "🔧 Individual Phases:"
	@echo "  make dataflow-cleanup   - Phase 0: Clean all Azure services"
	@echo "  make dataflow-ingest    - Phase 2: Ingest REAL data from data/raw/"
	@echo "  make dataflow-extract   - Phase 3: Extract knowledge to Cosmos DB"
	@echo "  make dataflow-query     - Phase 4: Test tri-modal search"
	@echo "  make dataflow-integrate - Phase 5: Integration testing"
	@echo "  make dataflow-advanced  - Phase 6: GNN training & deployment"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v "^help:" | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# Setup & Development
setup: ## Install backend and frontend dependencies
	@echo "📦 Installing backend dependencies..."
	@pip install -r requirements.txt
	@echo "📦 Installing frontend dependencies..."
	@cd frontend && npm install
	@echo "✅ Setup completed"

setup-complete: setup fix-azure ## Complete setup including Azure services fix
	@echo "🚀 COMPLETE SETUP"
	@echo "================"
	@echo "✅ Dependencies installed"
	@echo "✅ Azure services configured"
	@echo "✅ Search index created"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run: make dataflow-full  (to populate data)"
	@echo "2. Run: make dev           (to start development servers)"

dev: ## Start development servers (API port 8000, Frontend port 5173)
	@echo "🚀 Starting development servers..."
	@echo "📍 Backend API: http://localhost:8000"
	@echo "📍 Frontend UI: http://localhost:5173"
	@make -j 2 dev-backend dev-frontend

dev-backend: ## Start backend API server
	@uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend: ## Start frontend dev server
	@cd frontend && npm run dev

# Health & Testing
health: ## Check system and Azure services health
	@echo "🔍 System Health Check"
	@echo "Checking Python environment..."
	@python -c "import sys; print(f'Python: {sys.version}')"
	@echo ""
	@echo "Checking agent imports..."
	@python -c "from agents.domain_intelligence.agent import domain_intelligence_agent; print('✅ Domain Intelligence Agent')"
	@python -c "from agents.knowledge_extraction.agent import knowledge_extraction_agent; print('✅ Knowledge Extraction Agent')"
	@python -c "from agents.universal_search.agent import universal_search_agent; print('✅ Universal Search Agent')"
	@echo ""
	@echo "Checking Azure connectivity..."
	@python -c "import asyncio; from agents.core.universal_deps import get_universal_deps; deps = asyncio.run(get_universal_deps()); print(f'✅ Azure services: {list(deps.get_available_services())}')"

test: ## Run tests with REAL Azure services (no mocks)
	@echo "🧪 Running tests with REAL Azure services..."
	@pytest -v --tb=short

test-unit: ## Run unit tests only
	@pytest -m unit -v

test-integration: ## Run integration tests
	@pytest -m integration -v

# Authentication & Environment
auth-check: ## Check Azure authentication status
	@echo "🔐 Azure Authentication Status"
	@./scripts/deployment/sync-auth.sh

auth-setup: ## Setup Azure authentication
	@echo "🔐 Setting up Azure authentication..."
	@az login
	@azd auth login
	@./scripts/deployment/sync-auth.sh validate

sync-env: ## Sync environment with azd
	@echo "🔄 Syncing environment..."
	@./scripts/deployment/sync-env.sh

fix-azure: ## Fix Azure authentication and setup all services
	@echo "🔧 Fixing Azure services authentication and setup..."
	@chmod +x scripts/fix-azure-services.sh
	@./scripts/fix-azure-services.sh
	@echo "✅ Azure services fixed. Run: make dataflow-full"

# Deployment
deploy: sync-env ## Deploy infrastructure only (fast)
	@echo "🚀 Deploying infrastructure..."
	@azd env set AUTO_POPULATE_DATA false
	@azd up --no-prompt
	@echo "✅ Infrastructure deployed"
	@echo "💡 Run 'make dataflow-full' to populate with data"

deploy-with-data: auth-check sync-env ## Deploy with full data pipeline
	@echo "🚀 Full deployment with data pipeline..."
	@./scripts/deployment/sync-auth.sh validate || (echo "❌ Authentication required. Run: make auth-setup" && exit 1)
	@azd env set AUTO_POPULATE_DATA true
	@azd up --no-prompt
	@echo "📥 Running data pipeline..."
	@make dataflow-full
	@echo "✅ Full deployment completed"

deploy-fast: deploy ## Alias for infrastructure-only deployment

# 6-Phase Data Pipeline - REAL data from data/raw/
dataflow-cleanup: ## Phase 0: Clean all Azure services
	@echo "🧹 Phase 0: Cleaning Azure services..."
	@python scripts/dataflow/phase0_cleanup/00_01_cleanup_azure_data.py
	@python scripts/dataflow/phase0_cleanup/00_02_cleanup_azure_storage.py
	@python scripts/dataflow/phase0_cleanup/00_03_verify_clean_state.py
	@echo "✅ Cleanup completed"

dataflow-validate: ## Phase 1: Validate agent connectivity
	@echo "🧪 Phase 1: Validating agent connectivity..."
	@python scripts/dataflow/phase1_validation/01_00_basic_agent_connectivity.py
	@echo "✅ Agent validation completed"

dataflow-ingest: ## Phase 2: Ingest REAL data from data/raw/
	@echo "📥 Phase 2: Ingesting REAL data..."
	@echo "Validating prerequisites..."
	@python scripts/dataflow/phase2_ingestion/02_00_validate_phase2_prerequisites.py || (echo "❌ Prerequisites failed" && exit 1)
	@echo "Uploading to storage..."
	@python scripts/dataflow/phase2_ingestion/02_02_storage_upload_primary.py
	@echo "Creating embeddings..."
	@python scripts/dataflow/phase2_ingestion/02_03_vector_embeddings.py
	@echo "Indexing for search..."
	@python scripts/dataflow/phase2_ingestion/02_04_search_indexing.py --source data/raw/
	@echo "✅ Data ingestion completed"

dataflow-extract: ## Phase 3: Extract knowledge to Cosmos DB
	@echo "🧠 Phase 3: Extracting knowledge..."
	@echo "Validating prerequisites..."
	@python scripts/dataflow/phase3_knowledge/03_00_validate_phase3_prerequisites.py
	@echo "Extracting entities..."
	@python scripts/dataflow/phase3_knowledge/03_01_basic_entity_extraction.py
	@echo "Storing in graph..."
	@python scripts/dataflow/phase3_knowledge/03_02_graph_storage.py
	@echo "Verifying extraction..."
	@python scripts/dataflow/phase3_knowledge/03_03_verification.py
	@echo "✅ Knowledge extraction completed"

dataflow-query: ## Phase 4: Test tri-modal search
	@echo "🔍 Phase 4: Testing query pipeline..."
	@python scripts/dataflow/phase4_query/04_01_query_analysis.py "How to train custom models with Azure AI?"
	@python scripts/dataflow/phase4_query/04_06_complete_query_pipeline.py "Azure AI services capabilities"
	@echo "✅ Query pipeline tested"

dataflow-integrate: ## Phase 5: Integration testing
	@echo "🔄 Phase 5: Integration testing..."
	@python scripts/dataflow/phase5_integration/05_01_full_pipeline_execution.py --verbose
	@python scripts/dataflow/phase5_integration/05_03_query_generation_showcase.py
	@echo "✅ Integration testing completed"

dataflow-advanced: ## Phase 6: GNN training and deployment
	@echo "🚀 Phase 6: Advanced features..."
	@echo "GNN async bootstrap..."
	@python scripts/dataflow/phase6_advanced/06_11_gnn_async_bootstrap.py
	@echo "GNN deployment pipeline..."
	@python scripts/dataflow/phase6_advanced/06_10_gnn_deployment_pipeline.py
	@echo "GNN training..."
	@python scripts/dataflow/phase6_advanced/06_01_gnn_training.py
	@echo "Configuration demo..."
	@python scripts/dataflow/phase6_advanced/06_03_config_system_demo.py
	@echo "✅ Advanced features completed"

dataflow-full: ## Complete 6-phase pipeline with REAL data
	@echo "🌊 EXECUTING COMPLETE 6-PHASE PIPELINE"
	@echo "====================================="
	@echo "Using REAL Azure services with REAL data from data/raw/"
	@echo ""
	@echo "🔐 Checking authentication..."
	@./scripts/deployment/sync-auth.sh validate || (echo "⚠️ Authentication may expire. Run: make auth-setup" && sleep 3)
	@echo ""
	@echo "Phase 0: Cleanup..."
	@make dataflow-cleanup
	@echo ""
	@echo "Phase 1: Validation..."
	@make dataflow-validate
	@echo ""
	@echo "Phase 2: Data Ingestion..."
	@make dataflow-ingest
	@echo ""
	@echo "Phase 3: Knowledge Extraction..."
	@make dataflow-extract
	@echo ""
	@echo "Phase 4: Query Pipeline..."
	@make dataflow-query
	@echo ""
	@echo "Phase 5: Integration..."
	@make dataflow-integrate
	@echo ""
	@echo "Phase 6: Advanced Features..."
	@make dataflow-advanced
	@echo ""
	@echo "🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY"

# Removed dataflow-no-cleanup - cleanup scripts now handle credentials properly

# Testing deployed services
test-search: ## Test tri-modal search locally
	@echo "🔍 Testing tri-modal search with REAL Azure services..."
	@python scripts/test_trimodal.py

test-backend: ## Test deployed backend health
	@echo "🌐 Testing deployed backend..."
	@curl -s "$$(./scripts/show-deployment-urls.sh | grep Backend | cut -d' ' -f2)/health" | jq . || echo "Backend not deployed"

test-backend-search: ## Test deployed backend search
	@echo "🔍 Testing deployed search API..."
	@curl -X POST "$$(./scripts/show-deployment-urls.sh | grep Backend | cut -d' ' -f2)/api/v1/search" \
		-H "Content-Type: application/json" \
		-d '{"query": "Azure machine learning", "max_results": 3}' | jq .

test-frontend: ## Test deployed frontend
	@echo "🖥️ Testing deployed frontend..."
	@curl -I "$$(./scripts/show-deployment-urls.sh | grep Frontend | cut -d' ' -f2)" | head -1

# Utility commands
clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning temporary files..."
	@rm -rf logs/ __pycache__/ .pytest_cache/
	@rm -rf scripts/dataflow/results/*.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup completed"

clean-all: clean dataflow-cleanup ## Complete cleanup including Azure services

logs: ## Show recent logs
	@echo "📋 Recent execution logs:"
	@ls -la logs/ 2>/dev/null || echo "No logs found"

status: ## Show deployment status
	@echo "📊 Deployment Status:"
	@./scripts/show-deployment-urls.sh 2>/dev/null || echo "Not deployed"
	@echo ""
	@make health

# Quick commands for common workflows
quick-test: dataflow-validate test-search ## Quick validation and search test

quick-setup: setup dataflow-ingest dataflow-extract ## Quick setup with data

quick-fix: fix-azure dataflow-full ## Fix Azure and run full pipeline

rebuild: clean setup ## Clean and rebuild

first-time-setup: setup-complete dataflow-full ## Complete first-time setup with data

.PHONY: help setup dev dev-backend dev-frontend health test test-unit test-integration \
        auth-check auth-setup sync-env deploy deploy-with-data deploy-fast \
        dataflow-cleanup dataflow-validate dataflow-ingest dataflow-extract \
        dataflow-query dataflow-integrate dataflow-advanced dataflow-full \
        test-search test-backend test-backend-search test-frontend \
        clean clean-all logs status quick-test quick-setup rebuild