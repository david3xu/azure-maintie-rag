# MaintIE Enhanced RAG - Simple Makefile
# Based on your existing scripts and workflow

.PHONY: help setup dev-install test clean build run docker-build docker-run data-setup

# Default target
help:
	@echo "MaintIE Enhanced RAG - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup      - Full project setup (calls your existing scripts)"
	@echo "  make dev-install - Development dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  make run        - Start API server"
	@echo "  make test       - Run all tests"
	@echo "  make test-unit  - Run unit tests only"
	@echo "  make clean      - Clean generated files"
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

# Development targets
run:
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test: test-unit test-integration

test-unit:
	PYTHONPATH=. ./venv/bin/pytest tests/test_azure_*.py tests/test_real_config.py tests/test_real_pipeline.py tests/test_real_vector_search.py --cov=src

test-integration:
	@echo "Starting API server for integration tests..."
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 & sleep 3
	PYTHONPATH=. ./venv/bin/pytest tests/test_real_api.py
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
