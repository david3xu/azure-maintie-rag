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

workflow-demo:
	@echo "🔄 Running Universal RAG workflow demonstration..."
	@echo "💡 This demonstrates text processing with workflow transparency"
	PYTHONPATH=. ./.venv/bin/python scripts/universal_rag_demo.py

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
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ htmlcov/ .coverage
	@echo "✅ Cache and temporary files cleaned"

clean-data:
	@echo "🧹 Cleaning processed data (preserving raw text)..."
	rm -rf data/processed/* data/indices/* data/cache/* data/metrics/* data/output/*
	@echo "✅ Processed data cleaned - raw text preserved"
	@echo "💡 Run 'make workflow-demo' to reprocess text through Universal RAG"

clean-all:
	@echo "🧹 Deep cleaning all generated files..."
	@$(MAKE) clean
	@$(MAKE) clean-data
	rm -rf .venv/
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
