# MaintIE Enhanced RAG - Complete Project Execution Guide

**Based on Real Codebase Analysis**

---

## 📋 **Project Overview**

**Architecture**: Clean Service Separation
- **Backend**: Complete API service (data + logic + API)
- **Frontend**: Pure UI consumer service
- **Deployment**: Docker Compose with health checks

**Technology Stack**:
- Backend: FastAPI + Azure OpenAI + FAISS + NetworkX + PyTorch
- Frontend: React 19.1 + TypeScript + Vite
- Deployment: Docker + GitHub Actions CI/CD

---

## 🚀 **Phase 1: Initial Setup & Environment**

### **1.1 Prerequisites**
```bash
# Required software
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- Git
- VSCode (recommended)
```

### **1.2 Repository Setup**
```bash
# Clone repository
git clone https://github.com/david3xu/azure-maintie-rag.git
cd azure-maintie-rag

# Check available commands
make help
```

### **1.3 Environment Configuration**
```bash
# Copy environment template
cp backend/config/environment_example.env .env

# Edit .env with your settings
# Required: Azure OpenAI credentials
OPENAI_API_KEY=your-azure-key
OPENAI_API_BASE=https://your-instance.openai.azure.com/
OPENAI_DEPLOYMENT_NAME=your-deployment
EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment
```

### **1.4 Full Project Setup**
```bash
# Complete setup (both services)
make setup

# What this does:
# - Creates Python virtual environment (backend/.venv)
# - Installs Python dependencies
# - Creates data directories (backend/data/{raw,processed,indices})
# - Installs Node.js dependencies for frontend
```

---

## 📊 **Phase 2: Data Preparation & Knowledge Extraction**

### **2.1 Universal RAG Data Setup (Text Files)**
```bash
# Universal RAG works directly with raw text files - no JSON format required!
# Simply place .txt or .md files in backend/data/raw/

# Examples of text files you can use:
# - Maintenance procedures (.txt)
# - Equipment manuals (.md)
# - Troubleshooting guides (.txt)
# - Technical documentation (.md)

# Check what text files you have:
ls backend/data/raw/*.txt backend/data/raw/*.md

# Universal RAG will automatically:
# ✅ Extract knowledge from any text
# ✅ Build vector indices with FAISS
# ✅ Create knowledge graphs
# ✅ Enable real-time workflow progress
```

### **2.2 Knowledge Extraction**
```bash
# Extract domain knowledge from data
cd backend
make knowledge

# What this does:
# - Processes raw MaintIE data
# - Builds knowledge graph (24,464 nodes, 17,412 edges)
# - Creates entity-document indexes
# - Generates domain knowledge configurations
```

### **2.3 Data Validation**
```bash
# Validate configuration
python config/validation.py

# Debug data structure
make debug-entities
```

---

## 🔧 **Phase 3: Development Setup**

### **3.1 Documentation Environment**
```bash
# Setup VSCode extensions for markdown
make docs-setup

# Check documentation status
make docs-status

# Preview documentation (if VSCode CLI available)
make docs-preview
```

### **3.2 Development Dependencies**
```bash
cd backend
make dev-install  # Adds pytest, black, isort, flake8
```

---

## 🧪 **Phase 4: Testing & Validation**

### **4.1 Unit Tests**
```bash
# Run all tests
make test

# Backend only
cd backend && make test-unit

# Specific test files
cd backend
PYTHONPATH=. pytest tests/test_real_config.py -v
PYTHONPATH=. pytest tests/test_real_pipeline.py -v
```

### **4.2 Integration Tests**
```bash
cd backend
make test-integration  # Starts API server and runs integration tests
```

### **4.3 Comprehensive System Tests**
```bash
cd backend
PYTHONPATH=. pytest tests/comprehensive_test_suite.py -v
```

### **4.4 Debug Scripts**
```bash
cd backend

# Debug all components
make debug-all

# Specific debugging
make debug-pipeline    # Pipeline issues
make debug-gnn        # GNN integration
make debug-entities   # Entity extraction
make debug-monitoring # Monitoring system
```

---

## 🏃 **Phase 5: Development Workflow**

### **5.1 Start Development Services**
```bash
# Both services
make dev
# Backend: http://localhost:8000
# Frontend: http://localhost:5174
# API Docs: http://localhost:8000/docs

# Individual services
make backend   # API service only
make frontend  # UI service only
```

### **5.2 Service Health Check**
```bash
# Check both services
make health

# Manual check
curl http://localhost:8000/api/v1/health
curl http://localhost:5174
```

### **5.3 API Endpoints (Available when running)**
```bash
# Health & Status
GET  /api/v1/health
GET  /api/v1/status
GET  /api/v1/metrics

# Query Processing
POST /api/v1/query/multi-modal
POST /api/v1/query/structured
POST /api/v1/query/comparison
```

---

## 🧠 **Phase 6: Advanced Features**

### **6.1 GNN Training Pipeline**
```bash
cd backend

# Run comprehensive GNN training
python scripts/train_comprehensive_gnn.py \
    --config scripts/example_comprehensive_gnn_config.json \
    --n_trials 10 \
    --k_folds 3

# Features: Hyperparameter optimization, cross-validation,
# experiment tracking, model checkpointing
```

### **6.2 GNN Integration Test**
```bash
cd backend
make debug-gnn  # Test GNN integration components
```

---

## 🐳 **Phase 7: Containerization**

### **7.1 Docker Development**
```bash
# Build and run with Docker Compose
make docker-up

# Stop containers
make docker-down

# Individual service builds
cd backend && make docker-build
cd frontend && docker build -t maintie-frontend .
```

### **7.2 Docker Health Checks**
```bash
# Built-in health checks in docker-compose.yml
# Backend: curl -f http://localhost:8000/api/v1/health
# Check: docker-compose ps
```

---

## 🚀 **Phase 8: Production Deployment**

### **8.1 Production Preparation**
```bash
cd backend

# Full production check
make prod-ready

# What this does:
# - Cleans all temporary files
# - Runs complete test suite
# - Builds Docker image
# - Validates API endpoints
```

### **8.2 Environment Variables (Production)**
```bash
# Key production settings in .env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Azure OpenAI (required)
OPENAI_API_KEY=your-production-key
OPENAI_API_BASE=your-production-endpoint

# Performance settings
MAX_QUERY_TIME=2.0
CACHE_TTL=3600
```

### **8.3 CI/CD Pipeline**
```yaml
# Automated via .github/workflows/ci.yml
# Triggers: push to main, develop, feature/**
# Actions:
# - Python 3.10 setup
# - Dependency installation
# - Full test suite
# - GNN training smoke test
# - Coverage reports
```

---

## 🔍 **Phase 9: Monitoring & Maintenance**

### **9.1 Runtime Monitoring**
```bash
# Check system metrics
curl http://localhost:8000/api/v1/metrics

# Monitor logs (if running in Docker)
docker-compose logs -f backend
docker-compose logs -f frontend
```

### **9.2 Cache Management**
```bash
# Clear processed data (if issues)
cd backend && make clean

# Force rebuild (rebuilds all caches)
cd backend
python -c "
from src.pipeline.enhanced_rag import MaintIEEnhancedRAG
rag = MaintIEEnhancedRAG()
rag.initialize_components(force_rebuild=True)
"
```

### **9.3 Performance Optimization**
```bash
# Performance analysis script (actual from codebase)
cd backend
python scripts/real_query_flow_script.py

# Output: Detailed algorithm timing and bottleneck analysis
```

---

## 📁 **Phase 10: Project Structure Understanding**

### **10.1 Key Directories**
```
backend/
├── api/                 # FastAPI endpoints
├── src/
│   ├── pipeline/        # RAG implementations
│   ├── knowledge/       # Data processing
│   ├── retrieval/       # Vector/graph search
│   ├── enhancement/     # Query analysis
│   ├── generation/      # LLM interface
│   ├── gnn/            # Graph neural networks
│   └── monitoring/      # Performance tracking
├── config/             # Settings & validation
├── scripts/            # Utility scripts
├── tests/              # Test suite
└── data/
    ├── raw/            # Original MaintIE data
    ├── processed/      # Transformed data
    └── indices/        # Search indexes

frontend/
├── src/
│   ├── components/     # React components
│   ├── services/       # API client
│   └── types/          # TypeScript types
└── public/             # Static assets
```

### **10.2 Key Files & Their Purpose**
```bash
# Configuration
backend/config/settings.py           # Centralized settings
backend/config/environment_example.env  # Environment template

# Main implementations
backend/src/pipeline/enhanced_rag.py     # Main orchestrator
backend/src/pipeline/rag_structured.py  # Optimized implementation
backend/src/pipeline/rag_multi_modal.py # Original implementation

# Data processing
backend/src/knowledge/data_transformer.py    # MaintIE data processing
backend/src/knowledge/entity_document_index.py  # Entity indexing

# API
backend/api/main.py                  # FastAPI application
backend/api/endpoints/               # API route handlers

# Automation
Makefile                            # Root commands
backend/Makefile                    # Backend commands
docker-compose.yml                  # Container orchestration
.github/workflows/ci.yml            # CI/CD pipeline
```

---

## 🎯 **Quick Reference Commands**

### **Daily Development**
```bash
make dev          # Start both services
make test         # Run all tests
make health       # Check service status
make clean        # Clean temporary files
```

### **Data Operations**
```bash
cd backend
make knowledge    # Extract domain knowledge
make debug-all    # Debug all components
make debug-pipeline  # Debug specific issues
```

### **Production**
```bash
make docker-up    # Start with Docker
make prod-ready   # Production validation
```

### **Troubleshooting**
```bash
cd backend
make debug-entities   # Debug entity extraction
make debug-gnn       # Debug GNN integration
python config/validation.py  # Validate config
```

---

## ⚠️ **Common Issues & Solutions**

### **1. Knowledge Graph Rebuilding**
```bash
# Issue: System rebuilds knowledge graph every time
# Solution: Check file paths in processed data
ls -la backend/data/processed/
# Should contain: knowledge_graph.json, maintenance_entities.json
```

### **2. Empty Entity Results**
```bash
# Issue: Entity-document index returns empty results
# Debug:
cd backend && make debug-entities
# Check entity extraction process step-by-step
```

### **3. Azure OpenAI Connection**
```bash
# Issue: API connection failures
# Validation:
cd backend
python tests/test_real_config.py
# Verify Azure endpoint and key configuration
```

### **4. Docker Issues**
```bash
# Issue: Container health checks failing
# Debug:
docker-compose ps
docker-compose logs backend
# Check backend/Dockerfile health check configuration
```

---

## 📚 **Documentation Resources**

- **Architecture**: `docs/` directory
- **API Reference**: http://localhost:8000/docs (when running)
- **Configuration**: `backend/config/environment_example.env`
- **Debugging**: `backend/debug/README.md`
- **Testing**: `backend/tests/` directory

---

**This guide is based on the actual codebase structure and real implementation files. All commands and procedures have been verified against the existing project structure.**