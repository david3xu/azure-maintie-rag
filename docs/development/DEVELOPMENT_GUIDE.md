# Development Guide

**Azure Universal RAG - Development Setup and Workflow**

## Quick Setup

### Prerequisites
- Python 3.11+ with pip
- Node.js 18+ with npm (for frontend)
- Azure CLI (for deployment)
- Git

### 5-Minute Start
```bash
# Clone and setup
git clone <repository>
cd azure-maintie-rag

# Setup services
make setup

# Start development
make dev
```

**URLs**:
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Frontend: http://localhost:5174

## Architecture Overview ✅ **NEW: CONFIG-EXTRACTION ARCHITECTURE**

### Two-Stage Intelligence System

The system now implements a sophisticated **Config-Extraction Architecture** with clear separation of concerns:

**Stage 1: Domain Intelligence Agent**
- Analyzes domain-wide patterns
- Generates optimized `ExtractionConfiguration`
- File: `agents/domain_intelligence_agent.py`

**Stage 2: Knowledge Extraction Agent**  
- Processes documents using configuration
- Produces structured `ExtractionResults`
- File: `agents/knowledge_extraction_agent.py`

**Orchestration Layer**
- Coordinates the two-stage workflow
- File: `agents/config_extraction_orchestrator.py`

### Testing the Config-Extraction Workflow

```bash
# Test the complete two-stage architecture
python test_config_extraction_workflow.py

# Expected output:
# ✅ Stage 1 Complete: ExtractionConfiguration generated
# ✅ Stage 2 Complete: Knowledge extraction completed  
# 🎉 ALL TESTS PASSED
```

### Key Interface Contract

```python
# Configuration passed from Domain Intelligence to Knowledge Extraction
from config.extraction_interface import ExtractionConfiguration, ExtractionResults

# Stage 1: Generate configuration
config = await domain_agent.generate_extraction_config(domain, file_path)

# Stage 2: Extract knowledge using configuration  
results = await extraction_agent.extract_knowledge_from_document(content, config)
```

## Development Workflow

### Backend Development
```bash
# Setup backend only
cd backend && make setup

# Run backend server
make run

# Run tests
make test

# Check code quality
make pre-commit
```

### Frontend Development
```bash
# Setup frontend
cd frontend && npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

### Testing
```bash
# Run all tests
make test

# Backend tests only
cd backend && python -m pytest tests/

# Architecture validation
cd backend && python tests/validation/validate_architecture.py
```

## Project Structure

```
azure-maintie-rag/
├── backend/           # FastAPI application
│   ├── api/          # API endpoints
│   ├── agents/       # AI agents and tools
│   ├── services/     # Business logic layer
│   ├── infra/        # Azure service clients
│   ├── config/       # Configuration management
│   └── tests/        # Test suites
├── frontend/         # React/TypeScript UI
├── infra/           # Azure Bicep templates
├── docs/            # Documentation
└── scripts/         # Root-level utilities
```

## Development Best Practices

### Code Quality
- **Pre-commit hooks**: Automatic formatting and linting
- **Type hints**: Required for all Python functions
- **Async patterns**: Use async/await throughout
- **Error handling**: Comprehensive try/catch with context

### Architecture Rules
- **Layer boundaries**: API → Services → Infrastructure
- **No circular imports**: Maintain dependency direction
- **Service abstraction**: Use dependency injection
- **Performance first**: Sub-3-second response guarantee

### Testing Strategy
- **Unit tests**: `tests/unit/` for individual components
- **Integration tests**: `tests/integration/` for Azure services
- **Validation tests**: `tests/validation/` for architecture compliance

## Configuration

### Environment Setup
```bash
# Development
cp backend/config/environments/development.env backend/.env

# Production
# Azure services configured via deployment
```

### Azure Services
- **Local development**: Uses Azure services in development mode
- **Testing**: Mock services available in `tests/fixtures/`
- **Production**: Managed identity authentication

## Troubleshooting

### Common Issues
1. **Import errors**: Check Python path and virtual environment
2. **Azure auth**: Run `az login` and check subscription
3. **Port conflicts**: Services use 8000 (backend) and 5174 (frontend)
4. **Pre-commit fails**: Run `make pre-commit` to fix formatting

### Debug Mode
```bash
# Backend with debug logging
PYTHONPATH=. python -m uvicorn api.main:app --reload --log-level debug

# View logs
tail -f backend/logs/current_session.log
```

### Performance Testing
```bash
# Load test API
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "domain": "general"}'
```

This guide covers essential development workflows. See `CODING_STANDARDS.md` for detailed code conventions.