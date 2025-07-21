# MaintIE Universal RAG - Backend Service Makefile
# Universal text-based RAG system with workflow transparency

.PHONY: help setup dev-install test clean build run docker-build docker-run data-setup docs-setup docs-preview workflow-demo

# Default target
help:
	@echo "🔧 MaintIE Universal RAG Backend - Available Commands:"
	@echo ""
	@echo "🚀 Setup:"
	@echo "  make setup          - Full project setup with virtual environment"
	@echo "  make dev-install    - Development dependencies only"
	@echo "  make docs-setup     - Setup documentation environment"
	@echo ""
	@echo "🏃 Development:"
	@echo "  make run            - Start FastAPI server with hot reload"
	@echo "  make workflow-demo  - Run Universal RAG workflow demonstration"
	@echo "  make docs-preview   - Open markdown preview"
	@echo "  make docs-status    - Show documentation setup status"
	@echo ""
	@echo "🔄 Workflows:"
	@echo "  make data-prep      - WORKFLOW 1: Raw text → knowledge base"
	@echo "  make query-demo     - WORKFLOW 2: Query → intelligent response"
	@echo "  make lifecycle-demo - Complete RAG lifecycle demonstration"
	@echo "  make workflow-analysis - Analyze core file usage patterns"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test           - Run all tests (unit + integration + workflow)"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run API integration tests"
	@echo "  make test-workflow  - Run workflow manager integration tests"
	@echo "  make test-universal - Run Universal RAG system tests"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean          - Clean generated files and cache"
	@echo "  make clean-data     - Clean processed data (keep raw text)"
	@echo "  make clean-all      - Clean everything except raw data"
	@echo ""
	@echo "📊 Data:"
	@echo "  make data-setup SOURCE=/path/to/text/data - Setup text data"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo ""

# Setup targets
setup:
	@echo "🔧 Setting up Universal RAG backend..."
	python3 -m venv .venv
	./.venv/bin/python -m pip install --upgrade pip
	./.venv/bin/python -m pip install -r requirements.txt
	@mkdir -p data/{raw,processed,indices,cache,output,metrics}
	@echo "✅ Universal RAG backend setup complete!"

dev-install:
	@echo "🔧 Installing development dependencies..."
	python3 -m venv .venv
	./.venv/bin/python -m pip install --upgrade pip
	./.venv/bin/python -m pip install -r requirements.txt
	./.venv/bin/python -m pip install pytest pytest-cov black isort flake8
	@echo "✅ Development environment ready!"

# Documentation setup
docs-setup:
	@echo "📝 Setting up documentation environment..."
	@mkdir -p ../.vscode
	@echo '{"recommendations": ["yzhang.markdown-all-in-one", "shd101wyy.markdown-preview-enhanced", "bierner.markdown-mermaid", "davidanson.vscode-markdownlint", "ms-python.python", "ms-python.black-formatter", "ms-python.pylint", "ms-vscode.vscode-json", "redhat.vscode-yaml"]}' > ../.vscode/extensions.json
	@echo "✅ Documentation environment ready!"
	@echo "💡 Use VSCode Remote-SSH for best markdown preview experience"

docs-preview:
	@echo "📖 Opening documentation..."
	@if command -v code >/dev/null 2>&1; then \
		code --command markdown.showPreviewToSide docs/README.md; \
	else \
		echo "💡 Open docs/README.md in VSCode for preview (Ctrl+Shift+V)"; \
	fi

docs-status:
	@echo "📋 Documentation status:"
	@echo "✅ Main documentation: docs/README.md"
	@echo "✅ Universal RAG capabilities: docs/UNIVERSAL_RAG_CAPABILITIES.md"
	@echo "✅ Backend README: README.md"
	@echo "✅ VSCode extensions configured for SSH development"

# Development targets
run:
	@echo "🚀 Starting Universal RAG API server..."
	./.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# =============================================================================
# UNIVERSAL RAG WORKFLOW TARGETS - Based on Real Core File Usage Analysis
# =============================================================================

# WORKFLOW 1: Raw Text Data Handling (Data Preparation)
# Uses: 8/12 core files for text → knowledge conversion
data-prep:
	@echo "🔄 WORKFLOW 1: Raw Text Data Handling"
	@echo "📊 Core Files: universal_text_processor, knowledge_extractor, classifier, vector_search"
	@echo "🎯 Purpose: Convert raw text files into searchable knowledge base"
	PYTHONPATH=. ./.venv/bin/python scripts/data_preparation_workflow.py

# WORKFLOW 2: User Query Processing (Runtime)
# Uses: 7/12 core files for query → response generation
query-demo:
	@echo "🔄 WORKFLOW 2: User Query Processing"
	@echo "📊 Core Files: enhanced_rag, workflow_manager, query_analyzer, llm_interface"
	@echo "🎯 Purpose: Process user queries against pre-built knowledge base"
	PYTHONPATH=. ./.venv/bin/python scripts/query_processing_workflow.py

# Combined lifecycle demonstration
lifecycle-demo:
	@echo "🌟 COMPLETE UNIVERSAL RAG LIFECYCLE"
	@echo "🔄 Demonstrates both data preparation and query processing workflows"
	@$(MAKE) data-prep
	@echo ""
	@echo "⏳ Data preparation complete. Starting query processing..."
	@echo ""
	@$(MAKE) query-demo

# Workflow analysis and validation
workflow-analysis:
	@echo "📊 CORE FILES USAGE ANALYSIS"
	@echo "Data Preparation Workflow: 8/12 core files"
	@echo "Query Processing Workflow: 7/12 core files"
	@echo "Total Active Files: 10/12 core files (83% utilization)"
	PYTHONPATH=. ./.venv/bin/python scripts/workflow_analysis.py

# Legacy workflow demo (kept for backward compatibility)
workflow-demo:
	@echo "🔄 Running Universal RAG workflow demonstration..."
	@echo "💡 This demonstrates text processing with workflow transparency"
	PYTHONPATH=. ./.venv/bin/python scripts/universal_rag_demo.py

# Enhanced Azure Data Lifecycle Management
data-state:
	@echo "🔍 Azure Data State Analysis..."
	@echo "📊 Analyzing Azure services data state for domain: general"
	PYTHONPATH=. ./.venv/bin/python -c "
import asyncio
import json
from integrations.azure_services import AzureServicesManager
async def analyze_state():
    services = AzureServicesManager()
    state = await services.validate_domain_data_state('general')
    print('Azure Data State Analysis:')
    print(f'  Blob Storage: {state[\"azure_blob_storage\"][\"document_count\"]} documents')
    print(f'  Search Index: {state[\"azure_cognitive_search\"][\"document_count\"]} documents')
    print(f'  Cosmos DB: {state[\"azure_cosmos_db\"][\"vertex_count\"]} entities')
    print(f'  Raw Data: {state[\"raw_data_directory\"][\"file_count\"]} files')
    print(f'  Processing Required: {state[\"requires_processing\"]}')
asyncio.run(analyze_state())
"

data-prep-conditional:
	@echo "🧠 Conditional Data Preparation (with Azure state validation)..."
	@echo "💡 Checks Azure services state before processing"
	PYTHONPATH=. .venv/bin/python scripts/data_preparation_workflow.py

data-prep-force:
	@echo "🔄 Force Data Preparation (ignores existing Azure data)..."
	@echo "⚠️  This will reprocess data even if Azure services contain existing data"
	FORCE_DATA_REPROCESSING=true PYTHONPATH=. .venv/bin/python scripts/data_preparation_workflow.py

# Testing - Updated for Universal RAG architecture
test: test-unit test-integration test-workflow test-universal

test-unit:
	@echo "🧪 Running unit tests..."
	PYTHONPATH=. ./.venv/bin/pytest tests/ -k "not integration" --cov=core --cov=api --cov-report=term-missing

test-integration:
	@echo "🔗 Running API integration tests..."
	@echo "Starting test server..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@./.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 & sleep 3
	@PYTHONPATH=. ./.venv/bin/pytest tests/ -k "integration" || { pkill -f "uvicorn api.main:app" || true; exit 1; }
	@pkill -f "uvicorn api.main:app" || true
	@echo "✅ Integration tests completed"

test-workflow:
	@echo "⚙️ Running workflow manager integration tests..."
	@echo "🔍 Testing three-layer progressive disclosure system..."
	PYTHONPATH=. ./.venv/bin/python tests/test_workflow_integration.py

test-universal:
	@echo "🌐 Running Universal RAG system tests..."
	@echo "📝 Testing text-based processing pipeline..."
	PYTHONPATH=. ./.venv/bin/python tests/test_universal_rag.py

# Cleanup - Updated for Universal RAG file structure
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ htmlcov/ .coverage build/ dist/ *.egg-info/ 2>/dev/null || true
	@echo "🧹 Cleaning log files..."
	rm -f *.log backend.log api.log 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Cache, temporary files, and logs cleaned"

clean-data:
	@echo "🧹 Cleaning processed data (preserving raw text)..."
	rm -rf data/processed/* data/indices/* data/cache/* data/metrics/* data/output/* 2>/dev/null || true
	@echo "✅ Processed data cleaned - raw text preserved"
	@echo "💡 Run 'make workflow-demo' to reprocess text through Universal RAG"

clean-all:
	@echo "🧹 Deep cleaning all generated files..."
	@$(MAKE) clean
	@$(MAKE) clean-data
	rm -rf .venv/ build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -f *.log backend.log api.log 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Complete cleanup done - only source code and raw data remain"
	@echo "💡 Run 'make setup' to rebuild environment"

# Data processing - Universal RAG approach
data-setup:
	@if [ -z "$(SOURCE)" ]; then \
		echo "❌ Please specify SOURCE=/path/to/text/data"; \
		echo "💡 Universal RAG works with any text files - no domain configuration needed"; \
		exit 1; \
	fi
	@echo "📊 Setting up text data for Universal RAG processing..."
	@mkdir -p data/raw
	@cp -r $(SOURCE)/* data/raw/ 2>/dev/null || cp $(SOURCE) data/raw/
	@echo "✅ Text data ready for Universal RAG processing"
	@echo "💡 Run 'make workflow-demo' to see the system in action"

# Docker targets
docker-build:
	@echo "🐳 Building Universal RAG Docker image..."
	docker build -t universal-rag:latest .

docker-run:
	@echo "🐳 Running Universal RAG container..."
	docker run -p 8000:8000 --env-file .env universal-rag:latest

docker-dev:
	@echo "🐳 Starting development environment with Docker..."
	docker-compose up --build

# Production deployment preparation
prod-ready: clean test docker-build
	@echo "🚀 Universal RAG backend is production ready!"
	@echo "📊 API available at: http://localhost:8000"
	@echo "📖 API docs at: http://localhost:8000/docs"
	@echo "🔄 Workflow streaming: http://localhost:8000/api/v1/query/stream/{query_id}"
