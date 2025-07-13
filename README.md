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
