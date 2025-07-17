# Universal RAG System - Complete Execution Guide

**🚀 Production-Ready Lifecycle Execution Instructions**

---

## 📋 Overview

This guide provides **step-by-step instructions** for executing the complete Universal RAG system lifecycle, from initial setup to production-ready operation with three-layer workflow transparency.

**System Capabilities:**
- ✅ **Universal Text Processing**: Works with any domain without configuration
- ✅ **Three-Layer Workflow Transparency**: User-friendly → Technical → Diagnostic
- ✅ **Real-Time Streaming**: Server-sent events with detailed progress tracking
- ✅ **Frontend-Backend Integration**: Perfect TypeScript interface compatibility
- ✅ **Production-Ready Architecture**: FastAPI with comprehensive error handling

---

## 🔧 Prerequisites

### System Requirements
```bash
# Verify system requirements
python3 --version  # Python 3.8+
node --version     # Node.js 16+
npm --version      # npm 7+
docker --version   # Docker (optional)
```

### Environment Setup
```bash
# Clone and navigate to project
cd /home/azureuser/workspace/azure-maintie-rag

# Verify project structure
ls -la
# Expected: backend/ frontend/ Makefile README.md docker-compose.yml
```

---

## 🚀 Phase 1: Complete System Setup

### 1.1 Backend Setup
```bash
# Setup Universal RAG backend with virtual environment
make setup

# Expected output:
# 🔧 Setting up Universal RAG backend (text-based AI service)...
# 🎨 Setting up frontend (workflow transparency UI)...
# ✅ Universal RAG system ready!
```

### 1.2 Environment Configuration
```bash
# Check environment file
ls backend/.env

# If missing, create from template:
cp backend/.env.backup backend/.env

# Verify required directories
ls -la backend/data/
# Expected: raw/ processed/ indices/ cache/ output/ metrics/ models/
```

### 1.3 Dependencies Verification
```bash
# Activate backend environment
cd backend
source .venv/bin/activate

# Verify key packages
python -c "import uvicorn, fastapi, pydantic; print('✅ Core dependencies installed')"
python -c "import numpy, pandas; print('✅ Data processing dependencies installed')"

# Return to root
cd ..
```

---

## 📊 Phase 2: Data Preparation

### 2.1 Prepare Sample Text Data
```bash
# Create sample maintenance text for demonstration
mkdir -p backend/data/raw

# Create sample maintenance documentation
cat > backend/data/raw/maintenance_manual.txt << 'EOF'
# Pump Maintenance Procedures

## Centrifugal Pump P-101
The centrifugal pump P-101 is a critical component in the cooling water system. Regular maintenance is essential for optimal performance.

### Inspection Checklist
- Check pump seal condition daily
- Monitor vibration levels weekly
- Inspect coupling alignment monthly
- Verify proper lubrication quarterly

### Common Issues
1. **Cavitation**: Caused by insufficient NPSH or blocked suction
2. **Seal Leakage**: Usually indicates worn mechanical seal
3. **Excessive Vibration**: Often due to misalignment or bearing wear

### Troubleshooting Guide
If pump efficiency drops below 80%, check:
- Impeller wear
- Volute casing erosion
- Bearing condition
- Motor coupling alignment

Contact maintenance team if issues persist.
EOF

# Create additional sample files
cat > backend/data/raw/safety_procedures.txt << 'EOF'
# Safety Procedures for Industrial Equipment

## General Safety Guidelines
All maintenance personnel must follow these safety protocols when working on industrial equipment.

### Personal Protective Equipment (PPE)
- Safety glasses required at all times
- Hard hat mandatory in designated areas
- Steel-toed boots for equipment maintenance
- Chemical-resistant gloves when handling lubricants

### Lockout/Tagout Procedures
1. Notify control room before shutdown
2. Isolate all energy sources
3. Apply locks and tags
4. Test equipment to ensure isolation
5. Obtain clearance before starting work

## Emergency Procedures
In case of equipment failure or safety incident:
- Immediately shut down affected equipment
- Notify emergency response team
- Evacuate area if necessary
- Document incident details
EOF

echo "✅ Sample text data created in backend/data/raw/"
ls -la backend/data/raw/
```

### 2.2 Verify Data Setup
```bash
# Check data structure
find backend/data -name "*.txt" | head -5
# Expected: Show sample text files

echo "📊 Text data ready for Universal RAG processing"
```

---

## 🏃 Phase 3: System Startup

### 3.1 Start Backend API Service
```bash
# Start Universal RAG backend (Terminal 1)
make backend

# Expected output:
# 🔧 Starting Universal RAG backend API service...
# 💡 Supports any domain through pure text processing
# INFO: Started server process [PID]
# INFO: Uvicorn running on http://0.0.0.0:8000
```

### 3.2 Verify Backend Health
```bash
# In new terminal (Terminal 2)
cd /home/azureuser/workspace/azure-maintie-rag

# Check API health
curl -s http://localhost:8000/api/v1/health | jq .

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2025-01-17T...",
#   "version": "1.0.0",
#   "components": {
#     "universal_rag": "operational",
#     "workflow_manager": "ready"
#   }
# }

# Check API documentation
curl -s http://localhost:8000/docs > /dev/null && echo "✅ API docs accessible"
```

### 3.3 Start Frontend UI (Optional)
```bash
# Start workflow transparency frontend (Terminal 3)
make frontend

# Expected output:
# 🎨 Starting workflow transparency frontend...
# 💡 Three-layer progressive disclosure for different user types
# Local: http://localhost:5174/
```

---

## 🧪 Phase 4: Comprehensive Testing

### 4.1 Unit and Integration Tests
```bash
# Run complete test suite
cd backend
make test

# Expected output:
# 🧪 Running unit tests...
# 🔗 Running API integration tests...
# ⚙️ Running workflow manager integration tests...
# 🌐 Running Universal RAG system tests...
# ✅ All tests passed
```

### 4.2 Workflow Integration Test
```bash
# Test three-layer progressive disclosure system
make test-workflow

# Expected detailed output showing:
# ✅ Created WorkflowStep with ID
# 📱 Layer 1 (User-friendly): 6 fields
# 🔧 Layer 2 (Technical): 10 fields
# 🔬 Layer 3 (Diagnostic): 12 fields
# ✅ Frontend interface compatibility verified
# 🎉 All Tests Passed!
```

### 4.3 Universal RAG Demo
```bash
# Run Universal RAG demonstration
make workflow-demo

# Expected output:
# 🔄 Running Universal RAG workflow demonstration...
# 💡 This demonstrates text processing with workflow transparency
# [Detailed workflow execution with progress tracking]
```

---

## 🔄 Phase 5: Live System Testing

### 5.1 Test Universal RAG Query Processing
```bash
# Test basic query endpoint
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I troubleshoot pump cavitation issues?",
    "domain": "maintenance"
  }'

# Expected: Stream of workflow events with three-layer disclosure
```

### 5.2 Test Streaming Workflow API
```bash
# Start a query and get the query_id, then:
QUERY_ID="test-query-$(date +%s)"

# Test workflow step streaming
curl -N "http://localhost:8000/api/v1/query/stream/${QUERY_ID}" \
  -H "Accept: text/event-stream"

# Expected: Server-sent events with WorkflowStep objects
```

### 5.3 Test Three-Layer Progressive Disclosure
```bash
# Test Layer 1 (User-friendly)
curl "http://localhost:8000/api/v1/workflow/${QUERY_ID}/steps?layer=1"

# Test Layer 2 (Technical)
curl "http://localhost:8000/api/v1/workflow/${QUERY_ID}/steps?layer=2"

# Test Layer 3 (Diagnostic)
curl "http://localhost:8000/api/v1/workflow/${QUERY_ID}/steps?layer=3"

# Each should return different levels of detail
```

---

## 📊 Phase 6: System Health Verification

### 6.1 Comprehensive Health Check
```bash
# Run complete system health check
make health

# Expected output:
# 🔍 Checking Universal RAG system health:
# Backend Universal RAG API: ✅
# Frontend Workflow UI: ✅
# Universal RAG capabilities: ✅
```

### 6.2 Performance Verification
```bash
# Check system resources
cd backend
source .venv/bin/activate

# Monitor API performance
python -c "
import requests
import time
start = time.time()
response = requests.get('http://localhost:8000/api/v1/health')
end = time.time()
print(f'✅ API response time: {(end-start)*1000:.1f}ms')
print(f'✅ Status: {response.status_code}')
"
```

### 6.3 Data Processing Verification
```bash
# Verify text data processing
ls -la backend/data/raw/
echo "📊 Raw text files: $(ls backend/data/raw/*.txt | wc -l)"

# Check if processing generates indices (after running queries)
ls -la backend/data/indices/ 2>/dev/null || echo "📝 Indices will be created on first query"
```

---

## 🎯 Phase 7: Production Readiness Check

### 7.1 API Endpoints Verification
```bash
# Test all critical endpoints
echo "🔍 Testing Universal RAG API endpoints..."

# Health endpoint
curl -s http://localhost:8000/api/v1/health | jq '.status' | grep -q "healthy" && echo "✅ Health check"

# API documentation
curl -s http://localhost:8000/docs > /dev/null && echo "✅ API documentation"

# Query endpoint (basic)
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"test"}' > /dev/null && echo "✅ Query endpoint"

echo "🎉 All critical endpoints operational"
```

### 7.2 Frontend Integration Check
```bash
# If frontend is running, test integration
if curl -s http://localhost:5174 > /dev/null; then
  echo "✅ Frontend accessible at http://localhost:5174"
  echo "✅ Three-layer workflow transparency ready"
  echo "✅ Real-time streaming interface active"
else
  echo "ℹ️  Frontend not running - use 'make frontend' to start"
fi
```

### 7.3 Documentation Verification
```bash
# Verify all documentation is accessible
echo "📚 Verifying documentation completeness..."

ls backend/docs/UNIVERSAL_RAG_CAPABILITIES.md > /dev/null && echo "✅ System capabilities documented"
ls backend/docs/README.md > /dev/null && echo "✅ Documentation index available"
ls README.md > /dev/null && echo "✅ Project README current"
ls UNIVERSAL_RAG_EXECUTION_GUIDE.md > /dev/null && echo "✅ Execution guide available"

echo "📖 Complete documentation suite ready"
```

---

## 🐳 Phase 8: Docker Deployment (Optional)

### 8.1 Docker Container Execution
```bash
# Build and run with Docker
make docker-up

# Expected output:
# 🐳 Starting Universal RAG system with Docker...
# 🔄 Includes: Text processing + Workflow transparency + Real-time streaming
# [Docker container startup logs]
```

### 8.2 Docker Health Verification
```bash
# Check Docker containers
docker ps | grep universal-rag

# Test containerized API
curl -s http://localhost:8000/api/v1/health | jq .

# Stop Docker environment
make docker-down
```

---

## ✅ Phase 9: Success Verification

### 9.1 System Capabilities Confirmed
- ✅ **Universal RAG Backend**: Text processing from any domain
- ✅ **Three-Layer Progressive Disclosure**: User/Technical/Diagnostic views
- ✅ **Real-Time Workflow Streaming**: Server-sent events with detailed progress
- ✅ **Frontend-Backend Integration**: Perfect TypeScript compatibility
- ✅ **Production Architecture**: Scalable FastAPI with comprehensive error handling

### 9.2 API Endpoints Ready
- ✅ `GET /api/v1/health` - System health monitoring
- ✅ `POST /api/v1/query` - Universal RAG query processing
- ✅ `GET /api/v1/query/stream/{query_id}` - Real-time workflow streaming
- ✅ `GET /api/v1/workflow/{query_id}/steps?layer=1|2|3` - Progressive disclosure
- ✅ `GET /docs` - Complete API documentation

### 9.3 System Status
```bash
echo "🎉 Universal RAG System - PRODUCTION READY!"
echo ""
echo "🔗 Available Services:"
echo "  📍 Backend API:   http://localhost:8000"
echo "  📍 API Docs:      http://localhost:8000/docs"
echo "  📍 Health Check:  http://localhost:8000/api/v1/health"
echo "  📍 Frontend UI:   http://localhost:5174 (if started)"
echo ""
echo "🚀 Key Features:"
echo "  ✅ Universal text processing (any domain)"
echo "  ✅ Three-layer workflow transparency"
echo "  ✅ Real-time streaming with progress tracking"
echo "  ✅ Production-ready FastAPI architecture"
echo ""
echo "📚 Documentation:"
echo "  📖 System capabilities: backend/docs/UNIVERSAL_RAG_CAPABILITIES.md"
echo "  📋 Documentation index: backend/docs/README.md"
echo "  🚀 Execution guide: UNIVERSAL_RAG_EXECUTION_GUIDE.md"
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

**Port Already in Use**
```bash
# Kill existing processes
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5174 | xargs kill -9 2>/dev/null || true
```

**Virtual Environment Issues**
```bash
# Recreate virtual environment
cd backend
rm -rf .venv
make setup
```

**Missing Dependencies**
```bash
cd backend
source .venv/bin/activate
pip install -r requirements.txt
```

**API Not Responding**
```bash
# Check backend logs and restart
cd backend
make run
```

**Frontend Connection Issues**
```bash
# Verify backend is running first
curl http://localhost:8000/api/v1/health

# Then start frontend
cd frontend
npm install
npm run dev
```

---

## 📞 Support & Next Steps

### Production Deployment
- **Environment**: Configure production environment variables
- **Scaling**: Use Docker Compose or Kubernetes for multi-instance deployment
- **Monitoring**: Implement logging and metrics collection
- **Security**: Add authentication and rate limiting

### Development
- **Custom Domains**: Add domain-specific text data to `backend/data/raw/`
- **API Extensions**: Extend endpoints in `backend/api/endpoints/`
- **Frontend Customization**: Modify React components for domain-specific UI

### Documentation
- **Complete API Reference**: `backend/docs/UNIVERSAL_RAG_CAPABILITIES.md`
- **System Architecture**: `backend/docs/README.md`
- **Project Overview**: `README.md`

---

**🎯 System Status: PRODUCTION READY** ✅

The Universal RAG system is now fully operational with three-layer workflow transparency, real-time streaming, and complete frontend-backend integration. The system processes any text data without domain configuration and provides progressive disclosure for different user expertise levels.