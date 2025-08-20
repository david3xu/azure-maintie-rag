# Azure Universal RAG - Simplified Makefile
# Core functionality only, no complex session management

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Basic Configuration
PYTHONPATH := $(PWD)
USE_MANAGED_IDENTITY := false

help: ## Show available commands
	@echo "🚀 Azure Universal RAG - Simple Commands"
	@echo "========================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Essential Commands
health: ## Check system health
	@echo "🔍 System Health Check"
	@PYTHONPATH=$(PYTHONPATH) python -c "import asyncio; from agents.core.universal_deps import get_universal_deps; asyncio.run(get_universal_deps())"

deploy: ## Deploy to Azure (simple)
	@echo "🚀 Deploying to Azure..."
	@azd up

populate-data: ## Populate data in Azure services
	@echo "📥 Populating data..."
	@PYTHONPATH=$(PYTHONPATH) USE_MANAGED_IDENTITY=$(USE_MANAGED_IDENTITY) python scripts/dataflow/phase2_ingestion/02_02_storage_upload_primary.py
	@PYTHONPATH=$(PYTHONPATH) USE_MANAGED_IDENTITY=$(USE_MANAGED_IDENTITY) python scripts/dataflow/phase2_ingestion/02_03_vector_embeddings.py
	@PYTHONPATH=$(PYTHONPATH) USE_MANAGED_IDENTITY=$(USE_MANAGED_IDENTITY) python scripts/dataflow/phase2_ingestion/02_04_search_indexing.py --source data/raw --domain discovered_content

extract-knowledge: ## Extract entities and relationships
	@echo "🧠 Extracting knowledge..."
	@PYTHONPATH=$(PYTHONPATH) USE_MANAGED_IDENTITY=$(USE_MANAGED_IDENTITY) python scripts/dataflow/phase3_knowledge/03_01_basic_entity_extraction.py
	@PYTHONPATH=$(PYTHONPATH) USE_MANAGED_IDENTITY=$(USE_MANAGED_IDENTITY) python scripts/dataflow/phase3_knowledge/03_02_graph_storage.py

test-search: ## Test enhanced tri-modal search
	@echo "🔍 Testing enhanced search..."
	@PYTHONPATH=$(PYTHONPATH) python -c "import asyncio; from agents.orchestrator import UniversalOrchestrator; asyncio.run(UniversalOrchestrator().process_full_search_workflow('azure language training', max_results=3))"

test-backend: ## Test deployed backend
	@echo "🌐 Testing deployed backend..."
	@curl -s "https://ca-backend-maintie-rag-prod.graymeadow-1e9c52ba.westus2.azurecontainerapps.io/health" | jq .

test-frontend: ## Check frontend accessibility
	@echo "🖥️  Testing frontend..."
	@curl -I "https://ca-frontend-maintie-rag-prod.graymeadow-1e9c52ba.westus2.azurecontainerapps.io"

full-setup: populate-data extract-knowledge ## Complete data setup
	@echo "✅ Full setup completed"

clean: ## Clean logs and temporary files
	@echo "🧹 Cleaning..."
	@rm -rf logs/
	@rm -rf scripts/dataflow/results/

# Development
dev-backend: ## Start backend development server
	@echo "🔧 Starting backend..."
	@uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend: ## Start frontend development server
	@echo "🎨 Starting frontend..."
	@cd frontend && npm run dev

# Legacy support (redirect to simple commands)
dataflow-full: full-setup ## Legacy: Use full-setup instead
dataflow-incremental: populate-data extract-knowledge ## Legacy: Use populate-data + extract-knowledge
dataflow-ingest: populate-data ## Legacy: Use populate-data
dataflow-extract: extract-knowledge ## Legacy: Use extract-knowledge

.PHONY: help health deploy populate-data extract-knowledge test-search test-backend test-frontend full-setup clean dev-backend dev-frontend dataflow-full dataflow-incremental dataflow-ingest dataflow-extract
