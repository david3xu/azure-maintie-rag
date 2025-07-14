# MaintIE Enhanced RAG

Enterprise-Ready Maintenance Intelligence Backend

---

## 🚀 Overview

MaintIE Enhanced RAG is a production-grade backend system for advanced maintenance intelligence, combining:

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

- Create Python virtual environments for the backend.
- Install all Python dependencies for the backend.
- Create necessary data directories within `backend/data/`.
- Install Node.js dependencies for the frontend.

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

## 📝 Documentation

- See `docs/` for architecture, configuration, and usage guides.
- See `backend/scripts/extract_knowledge.py` for knowledge extraction from MaintIE data.

---

## 🔬 Comprehensive GNN Training Pipeline

MaintIE RAG includes a **research-level, end-to-end GNN training pipeline** for advanced experimentation and publication-ready results.

**Features:**
- Hyperparameter optimization (Optuna)
- Cross-validation (k-fold)
- Advanced training: schedulers, early stopping, gradient clipping, label smoothing, class weighting
- Comprehensive evaluation: accuracy, precision, recall, F1, AUC, confusion matrix, per-class analysis
- Ablation studies
- Experiment tracking (Weights & Biases)
- Model checkpointing and result saving

**How to Use:**
- **CLI:**
  ```bash
  python backend/scripts/train_comprehensive_gnn.py --config backend/scripts/example_comprehensive_gnn_config.json --n_trials 10 --k_folds 3
  python backend/scripts/train_comprehensive_gnn.py  # uses default config
  ```
- **Config:** Edit `backend/scripts/example_comprehensive_gnn_config.json` or provide your own.
- **API:** Import and call `run_comprehensive_gnn_training()` from `src.gnn.comprehensive_trainer`.

**Documentation:**
- See `backend/scripts/README_comprehensive_gnn.md` for CLI/config details.
- See module docstring in `backend/src/gnn/comprehensive_trainer.py` for full feature list and integration points.

**CI/CD:**
- The pipeline is smoke-tested in CI to ensure research code health.

---

## 🐳 Docker

To build and run both backend and frontend services using Docker:

```bash
make docker-up
```

---

## 📂 Project Structure

- `backend/` - Complete Backend API service (includes `data/`, `src/`, `api/`, `config/`, `scripts/`)
  - `backend/data/` - Raw, processed data, and indices.
- `frontend/` - Pure UI consumer service
- `docs/` - Documentation
- `.env` - Environment variables (copy from `.env.example`)
- `docker-compose.yml` - Docker Compose configuration for both services
- `Makefile` - Root Makefile for orchestrating both services

---

## 🤝 Contributing

PRs and issues welcome!

---

## 📄 License

See `LICENSE` (add your license here).
