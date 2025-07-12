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

---

## ✨ Features

- MaintIE data processing and knowledge graph construction
- Advanced query analysis and concept expansion
- Multi-modal retrieval (vector/entity/graph)
- LLM-powered, safety-aware response generation
- Configurable domain knowledge (JSON)
- Docker and virtualenv support
- Health, metrics, and system status endpoints

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/david3xu/azure-maintie-rag.git
cd azure-maintie-rag
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

- Copy `.env.example` to `.env` and set your OpenAI API key and other settings.
- Place MaintIE data files in `data/raw/` (see below).

### 5. Prepare MaintIE data

- Place `gold_release.json` and `silver_release.json` in `data/raw/`.
- Run the data transformer or use the provided scripts to process data.

---

## 🚦 Quick Start

### Run the API server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Test the API

```bash
curl http://localhost:8000/api/v1/health
```

---

## 📝 Documentation

- See `docs/` for architecture, configuration, and usage guides.
- See `scripts/extract_knowledge.py` for knowledge extraction from MaintIE data.

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## 📂 Project Structure

- `src/` - Core backend modules
- `api/` - FastAPI app and endpoints
- `config/` - Settings and domain knowledge config
- `data/` - Raw and processed data
- `scripts/` - Utility scripts
- `docs/` - Documentation

---

## 🤝 Contributing

PRs and issues welcome!

---

## 📄 License

See `LICENSE` (add your license here).
