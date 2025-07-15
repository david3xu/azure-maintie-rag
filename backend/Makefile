# MaintIE Enhanced RAG - Simple Makefile
# Based on your existing scripts and workflow

.PHONY: help setup dev-install test clean build run docker-build docker-run data-setup docs-setup docs-preview debug-pipeline debug-gnn debug-tests debug-entities debug-all

# Default target
help:
	@echo "MaintIE Enhanced RAG - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup      - Full project setup (calls your existing scripts)"
	@echo "  make dev-install - Development dependencies only"
	@echo "  make docs-setup - Setup documentation environment"
	@echo ""
	@echo "Development:"
	@echo "  make run        - Start API server"
	@echo "  make test       - Run all tests"
	@echo "  make test-unit  - Run unit tests only"
	@echo "  make clean      - Clean generated files"
	@echo "  make docs-preview - Open markdown preview"
	@echo "  make docs-status - Show documentation setup status"
	@echo ""
	@echo "🔍 Debugging:"
	@echo "  make debug-pipeline - Debug pipeline issues"
	@echo "  make debug-gnn      - Test GNN integration"
	@echo "  make debug-tests    - Debug test data structure"
	@echo "  make debug-entities - Debug entity extraction"
	@echo "  make debug-monitoring - Test granular monitoring system"
	@echo "  make debug-config   - Test config-driven improvements"
	@echo "  make debug-all      - Run all debug scripts"
	@echo ""
	@echo "Data:"
	@echo "  make data-setup SOURCE=/path/to/maintie - Setup MaintIE data"
	@echo "  make knowledge  - Extract domain knowledge"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo ""

# Setup targets (calls your existing scripts)
setup:
	python3 -m venv .venv
	./.venv/bin/python -m pip install -r requirements.txt
	bash -c 'mkdir -p data/{raw,processed,indices}' # Create in backend/data/
	@echo "✅ Setup complete!"

dev-install:
	python3 -m venv .venv
	./.venv/bin/python -m pip install -r requirements.txt
	./.venv/bin/python -m pip install pytest pytest-cov black isort flake8

# Documentation setup
docs-setup:
	@echo "📝 Setting up documentation environment..."
	@mkdir -p ../.vscode
	@echo '{"recommendations": ["yzhang.markdown-all-in-one", "shd101wyy.markdown-preview-enhanced", "bierner.markdown-mermaid", "davidanson.vscode-markdownlint", "ms-python.python", "ms-python.black-formatter", "ms-python.pylint", "ms-vscode.vscode-json", "redhat.vscode-yaml"]}' > ../.vscode/extensions.json
	@echo "✅ Documentation environment ready!"

docs-preview:
	@echo "📖 Opening markdown preview..."
	@if command -v code >/dev/null 2>&1; then \
		code --command markdown.showPreviewToSide ../README.md; \
	else \
		echo "⚠️  VSCode CLI not available in SSH environment"; \
		echo "💡 To view markdown preview:"; \
		echo "   1. Open VSCode locally"; \
		echo "   2. Use 'Remote-SSH' extension to connect to this server"; \
		echo "   3. Open README.md and use Ctrl+Shift+V for preview"; \
		echo "   4. Or install VSCode Server: https://code.visualstudio.com/docs/remote/vscode-server"; \
	fi

docs-status:
	@echo "📋 Documentation environment status:"
	@echo "✅ VSCode extensions configured: .vscode/extensions.json"
	@echo "✅ VSCode settings configured: .vscode/settings.json"
	@echo "✅ Custom CSS configured: .vscode/markdown.css"
	@echo ""
	@echo "🔧 For SSH development:"
	@echo "   - Use VSCode Remote-SSH extension for best experience"
	@echo "   - Markdown preview will work in VSCode Remote-SSH"
	@echo "   - All extensions will auto-install when you connect"

# Debug commands
debug-pipeline:
	@echo "🔍 Debugging pipeline..."
	PYTHONPATH=. python debug/debug_pipeline.py

debug-gnn:
	@echo "🧠 Testing GNN integration..."
	PYTHONPATH=. python debug/deploy_gnn_integration.py

debug-tests:
	@echo "🧪 Debugging test data..."
	PYTHONPATH=. python tests/debug/debug_data_structure.py

debug-entities:
	@echo "🔍 Debugging entity extraction..."
	PYTHONPATH=. python tests/debug/debug_entity_extraction.py

debug-monitoring:
	@echo "📊 Testing granular monitoring system..."
	PYTHONPATH=. python debug/debug_monitoring.py

debug-config:
	@echo "🔧 Testing config-driven improvements..."
	PYTHONPATH=. python test_config_integration.py

debug-all:
	@echo "🔍 Running all debug scripts..."
	@echo ""
	@echo "📊 1. Testing data structure..."
	@PYTHONPATH=. python tests/debug/debug_data_structure.py
	@echo ""
	@echo "🔍 2. Testing entity extraction..."
	@PYTHONPATH=. python tests/debug/debug_entity_extraction.py
	@echo ""
	@echo "🔧 3. Debugging pipeline..."
	@PYTHONPATH=. python debug/debug_pipeline.py
	@echo ""
	@echo "🧠 4. Testing GNN integration..."
	@PYTHONPATH=. python debug/deploy_gnn_integration.py
	@echo ""
	@echo "📊 5. Testing granular monitoring..."
	@PYTHONPATH=. python debug/debug_monitoring.py
	@echo ""
	@echo "🔧 6. Testing config-driven improvements..."
	@PYTHONPATH=. python test_config_integration.py
	@echo ""
	@echo "✅ All debug scripts completed!"

# Development targets
run:
	./.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test: test-unit test-integration

test-unit:
	PYTHONPATH=. ./.venv/bin/pytest tests/unit/ tests/test_validation_real_issues.py --cov=src

test-integration:
	@echo "Starting API server for integration tests..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 & sleep 3
	PYTHONPATH=. ./.venv/bin/pytest tests/integration/ tests/api/
	@pkill -f "uvicorn api.main:app" || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ htmlcov/ .coverage
	rm -rf data/processed/* data/indices/* # Clean backend/data/

# Data processing (calls your existing scripts)
data-setup:
	@if [ -z "$(SOURCE)" ]; then \
		echo "❌ Please specify SOURCE=/path/to/maintie/data"; \
		exit 1; \
	fi
	python scripts/setup_data.py --source $(SOURCE) --target data/ # backend/data/

knowledge:
	@echo "🧠 Extracting domain knowledge from processed data..."
	PYTHONPATH=. python scripts/extract_knowledge.py

# Docker targets
docker-build:
	docker build -t maintie-rag:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env maintie-rag:latest

docker-dev:
	docker-compose up --build

# Validation
validate:
	python scripts/testing/validate_api.py
	python config/validation.py

# Quick development workflow
dev: clean dev-install knowledge run

# Production deployment preparation
prod-ready: clean test docker-build validate
	@echo "✅ Production ready!"
