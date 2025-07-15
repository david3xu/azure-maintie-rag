# MaintIE Enhanced RAG

**Enterprise-Ready Maintenance Intelligence Backend**

---

## 🚀 Overview

MaintIE Enhanced RAG is a **production-grade backend system** for advanced maintenance intelligence, combining:

- **Knowledge graph extraction** from annotated maintenance data
- **Multi-modal retrieval** (vector, entity, and graph search)
- **Domain-aware LLM response generation**
- **Configurable domain knowledge** (no hard-coded rules)
- **FastAPI API** with health, metrics, and query endpoints
- **Clean Service Architecture** with a dedicated frontend UI

---

## ✨ Features

- MaintIE data processing and knowledge graph construction
- Advanced query analysis and concept expansion
- Multi-modal retrieval (vector/entity/graph)
- LLM-powered, safety-aware response generation
- Configurable domain knowledge (JSON)
- Docker and virtualenv support
- Health, metrics, and system status endpoints
- Separated Backend API and Frontend UI services

---

## 🛠️ Technology Stack

```
Frontend Stack:
├─ React 19.1.0 + TypeScript 5.8.3
├─ Vite 7.0.4 (build tool)
├─ axios 1.10.0 (HTTP client)
└─ CSS custom styling

Backend Stack:
├─ FastAPI + uvicorn
├─ Azure OpenAI integration (openai>=1.13.3)
├─ FAISS 1.7.4 vector search
├─ NetworkX 3.2.0 graph processing
├─ PyTorch 2.0.0 + torch-geometric 2.3.0 (GNN)
├─ Optuna 3.0.0 + Weights & Biases 0.16.0 (experiment tracking)
└─ Comprehensive ML/AI pipeline
```

---

## 🚦 Quick Commands

This project uses a root `Makefile` to simplify common tasks for both backend and frontend services.

### Using Makefile

```bash
make help               # See all available commands
make setup              # Full project setup (backend and frontend)
make dev                # Start both backend API and frontend UI services
make backend            # Start backend API service only
make frontend           # Start frontend UI service only
make test               # Run all tests (backend and frontend)
make health             # Check health of both services
make docker-up          # Build and run Docker containers for both services via docker-compose
make docker-down        # Stop and remove Docker containers
make clean              # Clean generated files for both services
```

---

## 📝 Documentation Setup

### VSCode Environment (Recommended)

For the best development experience with enhanced markdown preview:

```bash
# From backend directory
make docs-setup    # Sets up VSCode environment with extensions
make docs-status   # Shows documentation setup status
make docs-preview  # Opens markdown preview (if VSCode CLI available)
```

**For SSH Development (Azure ML):**
- Use VSCode Remote-SSH extension for best experience
- All extensions auto-install when you connect
- Markdown preview works perfectly with `Ctrl+Shift+V`

**Configured Extensions:**
- Markdown All in One
- Markdown Preview Enhanced
- Markdown Mermaid
- Python, Black, Pylint
- JSON and YAML support

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/david3xu/azure-maintie-rag.git
cd azure-maintie-rag
```

### 2. Configure environment

- Copy `.env.example` to `.env` in the project root and set your OpenAI API key and other settings.

### 3. Full Project Setup

This command will:

- Create Python virtual environments for the backend
- Install all Python dependencies for the backend
- Create necessary data directories within `backend/data/`
- Install Node.js dependencies for the frontend

```bash
make setup
```

### 4. Prepare MaintIE Data (Optional)

If you have MaintIE data files (`gold_release.json` and `silver_release.json`), place them in `backend/data/raw/`. Then, run the data setup command:

```bash
make data-setup SOURCE=/path/to/your/maintie/data
```

(Adjust `SOURCE` path as needed. If your data is already in `backend/data/raw/`, you can omit `SOURCE` or point it there.)

---

## 🔬 Comprehensive GNN Training Pipeline

MaintIE RAG includes a **research-level, end-to-end GNN training pipeline** for advanced experimentation and publication-ready results.

### Features:
- Hyperparameter optimization (Optuna)
- Cross-validation (k-fold)
- Advanced training: schedulers, early stopping, gradient clipping, label smoothing, class weighting
- Comprehensive evaluation: accuracy, precision, recall, F1, AUC, confusion matrix, per-class analysis
- Ablation studies
- Experiment tracking (Weights & Biases)
- Model checkpointing and result saving

### How to Use:

**CLI:**
```bash
python backend/scripts/train_comprehensive_gnn.py \
    --config backend/scripts/example_comprehensive_gnn_config.json \
    --n_trials 10 \
    --k_folds 3

python backend/scripts/train_comprehensive_gnn.py  # uses default config
```

**Config:** Edit `backend/scripts/example_comprehensive_gnn_config.json` or provide your own.

**API:** Import and call `run_comprehensive_gnn_training()` from `src.gnn.comprehensive_trainer`.

### Documentation:
- See `backend/scripts/README_comprehensive_gnn.md` for CLI/config details
- See module docstring in `backend/src/gnn/comprehensive_trainer.py` for full feature list and integration points

### CI/CD:
- The pipeline is smoke-tested in CI to ensure research code health

---

## 🐳 Docker

To build and run both backend and frontend services using Docker:

```bash
make docker-up
```

---

## 📂 Project Structure

```
Project Root:
├─ backend/                    # Complete Backend API service
│  ├─ data/                   # Raw, processed data, and indices
│  ├─ src/                    # Core source code
│  ├─ api/                    # FastAPI endpoints
│  ├─ config/                 # Configuration files
│  ├─ scripts/                # Utility scripts
│  └─ tests/                  # Test suite
├─ frontend/                  # Pure UI consumer service
│  ├─ src/                    # React components
│  ├─ public/                 # Static assets
│  └─ package.json            # Node.js dependencies
├─ docs/                      # Documentation
├─ .vscode/                   # VSCode configuration
├─ .env                       # Environment variables
├─ docker-compose.yml         # Docker Compose configuration
└─ Makefile                   # Root Makefile for orchestrating services
```

---

## 🔄 Service Architecture

```
User Input → React Frontend → FastAPI Backend → Multi-modal RAG → GNN Processing → AI Response
     ↓             ↓              ↓               ↓               ↓              ↓
"pump failure"  handleSubmit()  POST /api/      Vector+Graph    GNN Model      JSON response
```

---

## 📚 Documentation

- See `docs/` for architecture, configuration, and usage guides
- See `backend/scripts/extract_knowledge.py` for knowledge extraction from MaintIE data
- Use `make docs-setup` for enhanced VSCode markdown experience

---

## 🤝 Contributing

PRs and issues welcome!

---

## 📄 License

See `LICENSE` (add your license here).
