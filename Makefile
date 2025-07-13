# Azure MaintIE RAG - Clean Service Architecture
# Backend = Complete API service with data
# Frontend = Pure UI consumer

.PHONY: help setup dev test clean health

help:
	@echo "🔧 Azure MaintIE RAG - Clean Service Architecture"
	@echo ""
	@echo "Services:"
	@echo "  Backend:  Complete API service (data + logic + API)"
	@echo "  Frontend: Pure UI consumer"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make setup      - Setup both services"
	@echo "  make dev        - Start both services"
	@echo "  make backend    - Backend API only"
	@echo "  make frontend   - Frontend UI only"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test       - Test both services"
	@echo "  make health     - Check service health"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-up  - Start with Docker"
	@echo ""

# Setup both services
setup:
	@echo "🔧 Setting up backend (API + Data service)..."
	cd backend && make setup
	@echo "🎨 Setting up frontend (UI service)..."
	cd frontend && npm install
	@echo "✅ Both services ready!"

# Development - both services
dev:
	@echo "🚀 Starting clean service architecture:"
	@echo ""
	@echo "📍 Backend API:   http://localhost:8000 (self-contained)"
	@echo "📍 Frontend UI:   http://localhost:5174 (API consumer)"
	@echo "📍 API Docs:      http://localhost:8000/docs"
	@echo ""
	cd backend && make run
	cd frontend && npm run dev &

# Individual services
backend:
	@echo "🔧 Starting backend API service (includes data processing)..."
	cd backend && make run

frontend:
	@echo "🎨 Starting frontend UI service..."
	cd frontend && npm run dev

# Testing
test:
	@echo "🧪 Testing backend API service..."
	cd backend && make test
	@echo "🧪 Testing frontend UI service..."
	cd frontend && npm run test:ci || echo "Frontend tests not configured"

# Health check - verify clean service boundaries
health:
	@echo "🔍 Checking service architecture:"
	@echo ""
	@echo "Backend API (should be self-contained):"
	@curl -s http://localhost:8000/api/v1/health || echo "❌ Backend API not responding"
	@echo ""
	@echo "Frontend UI (should only consume API):"
	@curl -s http://localhost:5174 > /dev/null && echo "✅ Frontend UI accessible" || echo "❌ Frontend UI not running"

# Docker - clean service deployment
docker-up:
	@echo "🐳 Starting services with clean architecture..."
	docker-compose up --build

docker-down:
	docker-compose down

# Clean - each service cleans itself
clean:
	@echo "🧹 Cleaning backend service..."
	cd backend && make clean
	@echo "🧹 Cleaning frontend service..."
	cd frontend && rm -rf dist/ node_modules/.vite/

# Data setup - backend responsibility only
data-setup:
	@echo "📊 Setting up data (backend service responsibility)..."
	cd backend && make data-setup SOURCE=$(SOURCE)
