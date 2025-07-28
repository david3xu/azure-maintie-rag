# Backend Directory Structure

Clean, organized structure for the Azure Universal RAG backend.

## 📂 Directory Organization

```
backend/
├── 📚 Core Documentation
│   ├── README.md                           # Main backend overview
│   ├── BACKEND_QUICKSTART.md              # Fast development setup
│   └── DOCUMENTATION_TABLE_OF_CONTENTS.md # Complete documentation index
│
├── ⚙️ Configuration & Setup
│   ├── .env                               # Environment variables (private)
│   ├── .flake8                           # Python linting configuration
│   ├── pyproject.toml                    # Python project configuration
│   ├── pytest.ini                       # Testing configuration
│   ├── requirements.txt                  # Python dependencies
│   ├── Dockerfile                        # Container configuration
│   └── Makefile                          # Build and development commands
│
├── 🏗️ Core Application Code
│   ├── api/                              # FastAPI endpoints and routes
│   ├── core/                             # Core business logic and Azure clients
│   ├── services/                         # High-level business services
│   ├── integrations/                     # External service integrations
│   └── utilities/                        # Shared utility functions
│
├── 📁 Data & Processing
│   ├── data/                             # Raw data, processed outputs, demos
│   ├── scripts/                          # Data processing and automation scripts
│   └── prompt_flows/                     # Azure Prompt Flow configurations
│
├── 🔧 Development & Operations
│   ├── config/                           # Application configuration files
│   ├── tests/                            # Test suite and test utilities
│   ├── logs/                             # Application logs and debug output
│   ├── outputs/                          # Generated models and results
│   └── venv/                             # Python virtual environment
│
└── 📖 Documentation
    └── docs/                             # Organized technical documentation
        ├── architecture/                 # GNN, ML, knowledge graph docs
        ├── demo/                         # Demo guides and API documentation
        ├── execution/                    # RAG lifecycle and execution reports
        └── core/                         # Codebase structure and cleanup docs
```

## 🎯 Key Features

### ✅ Clean Root Directory
- Only essential configuration and documentation files in root
- All technical docs organized in `/docs/` subdirectories
- Standalone scripts moved to `/scripts/`

### ✅ Logical Organization
- **Core business logic** in `/core/` and `/services/`
- **API layer** separated in `/api/`
- **Data processing** contained in `/data/` and `/scripts/`
- **Configuration** centralized in `/config/`

### ✅ Development-Friendly
- Clear separation of concerns
- Easy navigation and file discovery
- Proper documentation organization
- Standard Python project structure

---

*This structure supports both development productivity and production deployment.*