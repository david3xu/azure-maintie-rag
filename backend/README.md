# Universal RAG Backend

**Clean, organized backend structure for Universal RAG system**

## 📁 Directory Structure

```
backend/
├── 📄 .env                 # Environment variables
├── 📄 .env.backup          # Environment backup
├── 🐳 Dockerfile          # Container configuration
├── 🔧 Makefile            # Build and utility commands
├── 📦 pyproject.toml      # Python project configuration
├── 📋 requirements.txt    # Python dependencies
├── 📖 README.md           # This documentation
├── 🏗️ .venv/             # Virtual environment
│
├── 🚀 api/               # FastAPI REST endpoints
│   ├── endpoints/        # Individual endpoint files
│   │   ├── universal_query.py    # Universal RAG query endpoint
│   │   ├── health.py            # Health check endpoint
│   │   └── __init__.py
│   ├── models/           # API request/response models
│   │   └── query_models.py      # Query-related Pydantic models
│   ├── main.py           # FastAPI application entry point
│   └── __init__.py
│
├── ⚙️ config/            # Configuration files and settings
├── 🧠 core/              # Universal RAG core components
│   ├── classification/   # Text classification components
│   ├── enhancement/      # Text enhancement and processing
│   ├── extraction/       # Knowledge extraction modules
│   ├── generation/       # Response generation components
│   ├── gnn/              # Graph Neural Network components
│   ├── knowledge/        # Knowledge base management
│   ├── models/           # Core data models and schemas
│   ├── orchestration/    # Main RAG orchestration logic
│   │   ├── enhanced_rag_universal.py        # Enhanced Universal RAG
│   │   ├── universal_rag_orchestrator_complete.py  # Complete orchestrator
│   │   └── __init__.py
│   ├── retrieval/        # Document retrieval and search
│   ├── utilities/        # Core utility functions
│   └── workflow/         # Workflow management system
│       └── universal_workflow_manager.py    # Three-layer workflow transparency
│
├── 💾 data/              # Raw and processed data storage
│   └── models/           # Trained models and indices
├── 🐛 debug/             # Debug and development tools
├── 📚 docs/              # Documentation and guides
│   ├── README.md                           # Documentation index
│   ├── UNIVERSAL_RAG_CAPABILITIES.md       # Complete system capabilities
│   ├── FINAL_CONCISE_CODEBASE_SUMMARY.md   # Codebase summary
│   ├── UNIVERSAL_RAG_FINAL_STATUS.md       # Implementation status
│   ├── UNIVERSAL_RAG_CLEANUP_COMPLETED.md  # Cleanup documentation
│   ├── UNIVERSAL_RAG_CLEANUP_PLAN.md       # Cleanup planning
│   └── PHASE_3_UNIVERSAL_RAG_COMPLETION_SUMMARY.md  # Phase 3 summary
├── 🔗 integrations/      # External service integrations
│   ├── azure_openai.py   # Azure OpenAI integration
│   └── __init__.py
├── 📜 scripts/           # Utility and demo scripts
│   └── universal_rag_demo.py  # Universal RAG demonstration
├── 🧪 tests/             # All test files and suites
│   ├── test_workflow_integration.py  # Workflow system tests
│   ├── test_universal_rag.py         # Universal RAG tests
│   └── __init__.py
└── 🛠️ utilities/         # Shared utility functions
    ├── file_utils.py     # File handling utilities
    ├── logging.py        # Logging configuration
    ├── config_loader.py  # Configuration loading
    └── __init__.py
```

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run workflow integration tests
python tests/test_workflow_integration.py

# 4. Start Universal RAG API server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 System Health

- ✅ **Universal RAG Core**: Fully operational with three-layer workflow transparency
- ✅ **Domain-Agnostic**: Works with any text data without configuration
- ✅ **Test Coverage**: Comprehensive workflow and integration tests passing
- ✅ **Production Ready**: Clean architecture with real-time streaming
- ✅ **Frontend Integration**: Perfect TypeScript interface compatibility