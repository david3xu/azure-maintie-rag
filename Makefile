# Azure MaintIE Universal RAG - Clean Service Architecture
# Backend = Complete Universal RAG API service
# Frontend = Pure UI consumer with workflow transparency

.PHONY: help setup dev test clean clean-all clean-models clean-azure health docs-setup docs-status docs-preview azure-deploy azure-status azure-teardown

help:
	@echo "🔧 Azure MaintIE Universal RAG - Clean Service Architecture"
	@echo ""
	@echo "🏗️ Architecture:"
	@echo "  Backend:  Universal text-based RAG API (domain-agnostic)"
	@echo "  Frontend: React UI with three-layer workflow transparency"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make setup      - Setup both services"
	@echo "  make dev        - Start both services with hot reload"
	@echo "  make backend    - Backend Universal RAG API only"
	@echo "  make frontend   - Frontend UI only"
	@echo ""
	@echo "📝 Documentation:"
	@echo "  make docs-setup - Setup VSCode documentation environment"
	@echo "  make docs-status - Show documentation setup status"
	@echo "  make docs-preview - Open markdown preview"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test       - Test both services"
	@echo "  make test-workflow - Test Universal RAG workflow system"
	@echo "  make health     - Check service health"
	@echo ""
	@echo "🧹 Cleanup (preserves raw text data):"
	@echo "  make clean      - Clean processed data & build artifacts"
	@echo "  make clean-all  - Deep clean (preserves raw text only)"
	@echo "  make clean-models - Complete reset to source code"
	@echo "  make clean-azure - Clean Azure deployment artifacts"
	@echo ""
	@echo "🏗️ Azure Infrastructure:"
	@echo "  make azure-deploy   - Deploy all 10 Azure services"
	@echo "  make azure-status   - Check Azure infrastructure status"
	@echo "  make azure-teardown - Clean up Azure resources"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-up  - Start with Docker"
	@echo ""

# Setup both services
setup:
	@echo "🔧 Setting up Universal RAG backend (text-based AI service)..."
	cd backend && make setup
	@echo "🎨 Setting up frontend (workflow transparency UI)..."
	cd frontend && npm install
	@echo "✅ Universal RAG system ready!"
	@echo "💡 Backend works with ANY text data - no domain configuration needed"

# Development - both services
dev:
	@echo "🚀 Starting Universal RAG system:"
	@echo ""
	@echo "📍 Backend API:   http://localhost:8000 (Universal RAG service)"
	@echo "📍 Frontend UI:   http://localhost:5174 (Workflow transparency)"
	@echo "📍 API Docs:      http://localhost:8000/docs"
	@echo "📍 Workflow API:  http://localhost:8000/api/v1/query/stream/{query_id}"
	@echo ""
	@echo "🔄 Features: Three-layer progressive disclosure + real-time workflow tracking"
	@echo ""
	@cd backend && make run &
	@cd frontend && npm run dev

# Individual services
backend:
	@echo "🔧 Starting Universal RAG backend API service..."
	@echo "💡 Supports any domain through pure text processing"
	cd backend && make run

frontend:
	@echo "🎨 Starting workflow transparency frontend..."
	@echo "💡 Three-layer progressive disclosure for different user types"
	cd frontend && npm run dev

# Documentation setup
docs-setup:
	@echo "📝 Setting up documentation environment..."
	cd backend && make docs-setup

docs-status:
	@echo "📋 Checking Universal RAG documentation..."
	cd backend && make docs-status

docs-preview:
	@echo "📖 Opening Universal RAG documentation..."
	cd backend && make docs-preview

# Testing
test:
	@echo "🧪 Testing Universal RAG backend service..."
	cd backend && make test
	@echo "🧪 Testing workflow transparency frontend..."
	cd frontend && npm run test:ci || echo "Frontend tests not configured"

test-workflow:
	@echo "⚙️ Testing Universal RAG workflow system..."
	@echo "🔍 Testing three-layer progressive disclosure..."
	cd backend && make test-workflow

# Health check - verify Universal RAG system
health:
	@echo "🔍 Checking Universal RAG system health:"
	@echo ""
	@echo "Backend Universal RAG API:"
	@curl -s http://localhost:8000/api/v1/health || echo "❌ Backend API not responding"
	@echo ""
	@echo "Frontend Workflow UI:"
	@curl -s http://localhost:5174 > /dev/null && echo "✅ Frontend UI accessible" || echo "❌ Frontend UI not running"
	@echo ""
	@echo "Universal RAG capabilities:"
	@curl -s http://localhost:8000/docs > /dev/null && echo "✅ API documentation accessible" || echo "❌ API docs not available"

# Docker - Universal RAG system deployment
docker-up:
	@echo "🐳 Starting Universal RAG system with Docker..."
	@echo "🔄 Includes: Text processing + Workflow transparency + Real-time streaming"
	docker-compose up --build

docker-down:
	docker-compose down

# Enhanced cleanup for Universal RAG system
clean:
	@echo "🧹 Cleaning Universal RAG processed data..."
	@echo "  Preserves: Raw text data + trained models"
	@echo "  Cleans: Processed indices + cache + build artifacts"
	cd backend && make clean-data
	@echo "🧹 Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "🧹 Cleaning frontend build artifacts..."
	cd frontend && rm -rf dist/ node_modules/.vite/ .vite/ build/ 2>/dev/null || true
	@echo "🧹 Cleaning backend build artifacts..."
	cd backend && rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage 2>/dev/null || true
	@echo "✅ Clean complete! Raw text and models preserved."
	@echo "💡 Run 'make dev' to reprocess through Universal RAG pipeline"

clean-all:
	@echo "🧹 Deep cleaning Universal RAG system..."
	@echo "  Preserves: Raw text data only"
	@echo "  Cleans: All processed data + models + dependencies"
	cd backend && make clean-all
	@echo "🧹 Cleaning frontend completely..."
	cd frontend && rm -rf dist/ node_modules/.vite/ .vite/ build/ node_modules/ 2>/dev/null || true
	@echo "🧹 Cleaning all Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Deep clean complete! Only raw text data remains."
	@echo "💡 Run 'make setup && make dev' to rebuild Universal RAG system"

clean-models:
	@echo "🧹 COMPLETE RESET: Universal RAG system to source code..."
	@echo "  Preserves: Source code only"
	@echo "  Cleans: ALL data + models + dependencies + cache"
	cd backend && make clean-all
	cd frontend && rm -rf dist/ node_modules/.vite/ .vite/ build/ node_modules/ 2>/dev/null || true
	@echo "🧹 Cleaning all Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "htmlcov" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -type f -delete 2>/dev/null || true
	@echo "⚠️  COMPLETE RESET DONE! Only source code remains."
	@echo "💡 Run 'make setup && make dev' to rebuild from scratch"
	@echo "📝 Universal RAG works with any text data - no domain setup needed"

# Data setup - backend Universal RAG responsibility
data-setup:
	@echo "📊 Setting up text data for Universal RAG..."
	@echo "💡 Universal RAG processes any text files automatically"
	cd backend && make data-setup SOURCE=$(SOURCE)

# Azure Infrastructure Management
azure-deploy:
	@echo "🏗️ Deploying Azure Universal RAG infrastructure..."
	@echo "💡 Deploys all 10 Azure services (Storage, Search, ML, Cosmos DB, etc.)"
	./scripts/enhanced-complete-redeploy.sh

azure-status:
	@echo "📊 Checking Azure Universal RAG infrastructure status..."
	@echo "💡 Shows all 10 services and their operational status"
	./scripts/status-working.sh

azure-teardown:
	@echo "🧹 Cleaning up Azure Universal RAG infrastructure..."
	@echo "⚠️  This will delete ALL Azure resources in the resource group"
	@echo "💡 Use with caution - this action cannot be undone"
	./scripts/teardown.sh

clean-azure:
	@echo "🧹 Cleaning Azure deployment artifacts..."
	rm -f .deployment_token .deployment_storage_name .deployment_search_name .deployment_keyvault_name .deployment_cosmos_name 2>/dev/null || true
	@echo "✅ Azure deployment artifacts cleaned"
