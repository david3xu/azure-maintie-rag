# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 Critical Path (START HERE)

⚠️ **Always work from project root: `cd /workspace/azure-maintie-rag`**

### **30-Second Health Check (First Thing to Run)**
```bash
# Verify system is operational before making changes
export PYTHONPATH=/workspace/azure-maintie-rag
make health                               # Complete system + Azure status
python -c "from agents.core.universal_models import UniversalDomainAnalysis; print('✅ Core imports working')"
```

### **Most Common Development Workflow (90% of tasks)**
```bash
# 1. Quick validation (ALWAYS run first)
make dataflow-validate                    # Test all 3 PydanticAI agents (30 seconds)

# 2. Make code changes, then validate
black . && isort .                        # Format code
./scripts/hooks/pre-commit-domain-bias-check.sh  # Check for domain bias
pytest -m unit -x                        # Run unit tests, stop on first failure

# 3. Test with real data (when needed)
make dataflow-extract                     # Test knowledge extraction (2-3 minutes)
make dataflow-query                       # Test search functionality (1-2 minutes)
```

### **Emergency Commands (When Things Break)**
```bash
# Authentication issues (most common)
az login && azd auth login
./scripts/deployment/sync-auth.sh validate

# Import errors (second most common) 
export PYTHONPATH=/workspace/azure-maintie-rag
export OPENBLAS_NUM_THREADS=1

# Full system reset
make clean && make health
```

## Essential Commands

### Quick Start & Development
```bash
# Local Development
make setup                    # Install all dependencies (backend + frontend)
make dev                      # Start API (port 8000) + Frontend (port 5174)
make health                   # System health check with Azure service status

# Production Deployment (RECOMMENDED)
make deploy-with-data         # Full deployment with automated data pipeline

# Enterprise Authentication (University/Corporate Environments)
./scripts/deployment/sync-auth.sh                    # Diagnose authentication issues
./scripts/deployment/sync-auth.sh validate           # Validate auth before long operations
az login && azd auth login                           # Refresh tokens for enterprise AD

# Session Management (Enterprise Feature - Clean Log Replacement)
make session-report           # View current session with real-time Azure status
make clean                    # Clean current session and start fresh (preserves cumulative)
make health                   # Complete health check with comprehensive session reporting

# Easy Resource Cleanup (Cost Management)
./scripts/azd-down-fixed.sh --force --purge   # FIXED azd down - works with REAL deployment patterns
./scripts/azd-down-fixed.sh                   # Fixed azd down with confirmation prompts
# NOTE: azd down is fundamentally broken - generates NEW deployment names instead of finding existing ones

# Testing (REAL Azure services, NO mocks)
pytest                        # All tests with asyncio_mode=auto
pytest -m unit                # Agent logic tests
pytest -m integration         # Multi-service integration
pytest -m azure_validation    # Azure service health
pytest -x -vvv                # Stop on first failure, verbose
pytest -k "specific_test"     # Run specific test by pattern

# Code Quality (REQUIRED before commits)
black . --check && isort . --check-only         # Python formatting
cd frontend && npm run lint && npx tsc --noEmit # Frontend checks
./scripts/hooks/pre-commit-domain-bias-check.sh # Domain bias validation (ENFORCED)

# 6-Phase Data Pipeline
make dataflow-full            # Execute all phases (complete pipeline)
make dataflow-validate        # Phase 1: Validate 3 PydanticAI agents (CRITICAL)
make dataflow-extract         # Phase 3: Knowledge extraction
make dataflow-query           # Phase 4: Query & search testing

# Direct Script Execution (set environment variables)
OPENBLAS_NUM_THREADS=1 USE_MANAGED_IDENTITY=false PYTHONPATH=/workspace/azure-maintie-rag python <script>

# Azure Deployment
./scripts/deployment/sync-env.sh prod  # Switch environment & sync config
azd up                                  # Deploy Azure infrastructure
azd env set AUTO_POPULATE_DATA true    # Enable automated data pipeline
./scripts/show-deployment-urls.sh      # Get deployment URLs

# Container Building (when needed - usually automatic via azd)
az acr build --registry <acr-name> --image azure-maintie-rag/backend-prod:latest .
az acr build --registry <acr-name> --image azure-maintie-rag/frontend-prod:latest ./frontend
```

## Architecture Overview

**Production-ready multi-agent RAG system** with PydanticAI framework and **zero hardcoded domain bias**.

### Three Core PydanticAI Agents

1. **Domain Intelligence Agent** (`agents/domain_intelligence/agent.py:72-95`)
   - Discovers content characteristics dynamically (no predetermined categories)
   - Outputs: `UniversalDomainAnalysis` with vocabulary complexity, processing config
   - Entry: `run_domain_analysis()`, `domain_intelligence_agent`

2. **Knowledge Extraction Agent** (`agents/knowledge_extraction/agent.py:45-70`)
   - Extracts entities/relationships using Domain Intelligence output
   - Uses unified template: `infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2`
   - Entry: `run_knowledge_extraction()`, `knowledge_extraction_agent`

3. **Universal Search Agent** (`agents/universal_search/agent.py:50-75`)
   - Orchestrates mandatory tri-modal search: Vector + Graph + GNN
   - No fallback - all three modalities required (fail-fast design)
   - Entry: `run_universal_search()`, `universal_search_agent`

### Critical Architecture Files & Design Decisions

**Core Agent Framework:**
- `agents/core/universal_models.py:1-1536` - Domain-agnostic data structures (all Pydantic models)
  - **WHY**: Zero domain bias enforcement - models discover rather than classify
- `agents/core/agent_toolsets.py:1-200` - Centralized FunctionToolset management (REQUIRED pattern)  
  - **WHY**: Avoids Union type errors in PydanticAI 0.6.2, enables proper tool registration
- `agents/core/universal_deps.py:1-150` - Azure service dependency injection
  - **WHY**: Singleton pattern for Azure clients, handles authentication contexts automatically
- `agents/core/azure_pydantic_provider.py:1-80` - Azure OpenAI provider with managed identity
  - **WHY**: Seamless local dev vs Container Apps authentication switching
- `agents/orchestrator.py:1-300` - UniversalOrchestrator for agent coordination
  - **WHY**: Type-safe inter-agent communication with comprehensive error handling

**Agent Implementations:**
- `agents/domain_intelligence/agent.py:72-95` - Domain analysis agent definition
- `agents/knowledge_extraction/agent.py:45-70` - Entity/relationship extraction agent
- `agents/universal_search/agent.py:50-75` - Tri-modal search orchestration agent

**Infrastructure:**
- `infrastructure/prompt_workflows/templates/universal_knowledge_extraction.jinja2` - Unified extraction template
- `infrastructure/azure_ml/gnn_inference_client.py:1-200` - GNN model inference (fail-fast)
- `infrastructure/azure_auth/session_manager.py:1-150` - Authentication session management
- `config/azure_settings.py:1-200` - Centralized Azure configuration

**Testing & Quality:**
- `pytest.ini:20` - asyncio_mode=auto for PydanticAI agents
- `scripts/hooks/pre-commit-domain-bias-check.sh` - Domain bias enforcement

## Critical Development Rules

### 🚨 Zero Domain Bias (ENFORCED by pre-commit)

```python
# ❌ WRONG - Hardcoded domain categories
if domain in ["legal", "technical", "medical"]:
    complexity = 0.8

# ✅ CORRECT - Discover from content
complexity = measure_vocabulary_complexity(content)
parameters = adapt_based_on_measured_properties(complexity)
```

**Enforcement:** `scripts/hooks/pre-commit-domain-bias-check.sh` fails builds with violations

### 🏗️ PydanticAI Agent Pattern (REQUIRED)

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from agents.core.universal_deps import UniversalDeps

# 1. Create toolset FIRST
toolset = FunctionToolset()

# 2. Register tools on TOOLSET (not agent)
@toolset.tool  # ✅ CORRECT - use toolset.tool
async def analyze_content(ctx: RunContext[UniversalDeps], content: str):
    return await ctx.deps.azure_openai_client.analyze(content)

# 3. Create agent WITH toolset
agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    toolsets=[toolset],  # Pass toolset here
    deps_type=UniversalDeps,
    retries=3  # Fail-fast with limited retries
)
```

**⚠️ Using `@agent.tool` causes Union type errors with PydanticAI 0.6.2**

### 🚀 Fail-Fast Philosophy (STRICT ENFORCEMENT)

The system implements strict fail-fast principles:
- **NO fallback logic** when ANY service fails
- **NO fake success** responses ever
- **NO partial operation** - ALL THREE modalities required
- GNN endpoint returns **REAL failures** until model is ready
- System **FAILS COMPLETELY** if Vector, Graph, OR GNN unavailable
- Searches return **ERRORS** until ALL modalities operational
- This is **CORRECT behavior** - we fix issues, not bypass them

## Environment Variables

**Required for ALL Python operations:**
```bash
export PYTHONPATH=/workspace/azure-maintie-rag     # Critical for agent imports
export OPENBLAS_NUM_THREADS=1                     # GNN/PyTorch numerical stability
```

**Authentication Context (auto-detected):**
```bash
# Local Development (auto-detected when not in Container Apps)
export USE_MANAGED_IDENTITY=false          # Uses DefaultAzureCredential (az login)

# Azure Container Apps (auto-detected in cloud deployment)  
export USE_MANAGED_IDENTITY=true           # Uses ManagedIdentityCredential directly
```

**Common Usage Patterns:**
```bash
# Local development (most common)
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/workspace/azure-maintie-rag python <script>

# Direct agent testing
PYTHONPATH=/workspace/azure-maintie-rag python agents/domain_intelligence/agent.py

# Dataflow script execution
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_01_validate_domain_intelligence.py
```

## 6-Phase Data Pipeline

| Phase | Purpose | Key Scripts | Success Criteria |
|-------|---------|-------------|------------------|
| **0 - Cleanup** | Clean Azure data (preserves infrastructure) | `00_01_cleanup_all_services.py` | 0 documents/entities |
| **1 - Validation** | Test 3 PydanticAI agents (CRITICAL) | `01_0*_validate_*.py` | All agents respond |
| **2 - Ingestion** | Upload all docs from data/raw | `02_02_storage_upload.py` | All docs indexed |
| **3 - Extraction** | Build knowledge graph | `03_01_basic_entity_extraction.py` | 13+ entities, 8+ relationships |
| **4 - Query** | Test tri-modal search | `04_01_query_analysis.py` | 3+ results, 80%+ confidence |
| **5 - Integration** | End-to-end validation | `05_01_full_pipeline_execution.py` | Complete pipeline works |
| **6 - Advanced** | GNN training + async bootstrap | `06_11_gnn_async_bootstrap.py` | Model deployed to Azure ML |

**Data source:** `data/raw/` (Universal system adapts to any documents placed here)
**Quick run:** `make dataflow-full` (complete 6-phase pipeline with GNN bootstrap)

## Testing Strategy

**All tests use REAL Azure services (no mocks)**:

```bash
# Test execution with asyncio_mode=auto (pytest.ini:20)
pytest                        # All tests
pytest -m unit                # Agent logic
pytest -m integration         # Multi-service integration  
pytest -m azure_validation    # Service health
pytest -m performance         # Performance tests
pytest -x -vvv                # Debug mode (stop on first failure)

# Performance testing with timeout controls
OPENBLAS_NUM_THREADS=1 timeout 120 pytest -m performance -v

# Test specific dataflow phase
PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase1_validation/01_01_validate_domain_intelligence.py

# Run single test
pytest tests/test_agents.py::TestDomainIntelligenceAgent::test_agent_import -v
```

## Backend Development

**Backend-specific commands** (when working in backend directory):
```bash
cd backend  # Not present in current structure - use project root commands instead

# Alternative backend commands from project root:
PYTHONPATH=/workspace/azure-maintie-rag uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
PYTHONPATH=/workspace/azure-maintie-rag python -m pytest tests/ -v
```

**Note**: The project uses root-level commands primarily. Backend-specific Makefile commands referenced in session management may not be available.

## Common Issues & Solutions

### Critical Dependency Issues
| Issue | Solution | File Reference |
|-------|----------|----------------|
| PydanticAI Union Type Error | `pip install 'openai==1.98.0'` + use FunctionToolset pattern | `requirements.txt:1-20` |
| Import Errors | Always use `PYTHONPATH=/workspace/azure-maintie-rag` | All Python scripts |
| Agent Tool Registration Error | Use `@toolset.tool` not `@agent.tool` | `agents/core/agent_toolsets.py:1-200` |

### Azure Service Issues  
| Issue | Solution | Diagnostic Command |
|-------|----------|-------------------|
| Azure Authentication Failed | `az login && ./scripts/deployment/sync-env.sh prod` | `make health` |
| azd down fails with "no resources found" | Use fixed azd down: `./scripts/azd-down-fixed.sh --force --purge` | Deployment name mismatch |
| Agent Validation Failures | Test individual agent imports | See "Agent Import Issues" below |
| Search Returns 0 Results | Verify data ingestion completed | `make dataflow-validate` |
| GNN Training Fails | Check async bootstrap status | `scripts/dataflow/phase6_advanced/06_12_check_async_status.py` |

### Development Issues
| Issue | Solution | Prevention |
|-------|----------|------------|
| Domain Bias Failed | Remove hardcoded domains, use measured properties | `scripts/hooks/pre-commit-domain-bias-check.sh` |
| Test Timeouts | Use `timeout 120` and `OPENBLAS_NUM_THREADS=1` | Set in environment |
| Configuration Manager Errors | Import from `agents.core.simple_config_manager` | Never use `dynamic_config_manager` |
| Session Management Issues | Use `make clean` to reset session | Check `logs/session_report.md` |

## Session Management

Enterprise session tracking with clean log replacement (Makefile:14-60):
- **Clean Log Replacement**: Each session completely replaces previous logs (no accumulation)
- **Unique Session ID**: Generated per command (`YYYYMMDD_HHMMSS`)
- **Real-time Azure Status**: Azure service status captured in each session
- **Performance Tracking**: System metrics captured automatically
- **Cumulative Reporting**: Long-term session history preserved

**Session Files:**
- `logs/current_session` - Active session ID
- `logs/dataflow_execution_<SESSION_ID>.md` - Current session report
- `logs/azure_status_<SESSION_ID>.log` - Azure service status
- `logs/performance_<SESSION_ID>.log` - System performance metrics
- `logs/cumulative_dataflow_report.md` - Historical session data

```bash
make session-report  # View current session with real-time status
make clean          # Clean session and start fresh (preserves cumulative history)
make health         # Complete health check with comprehensive session reporting
```

## Development Workflow

```bash
# 1. Setup environment
cd /workspace/azure-maintie-rag
./scripts/deployment/sync-env.sh prod
export PYTHONPATH=/workspace/azure-maintie-rag

# 2. Validate agents (CRITICAL first step)
make dataflow-validate

# 3. Make changes and test
black . && isort .  # Format code
./scripts/hooks/pre-commit-domain-bias-check.sh
pytest -m unit

# 4. Test with real data
make dataflow-extract  # Test knowledge extraction
make dataflow-query    # Test search

# 5. Full validation
make dataflow-full  # Complete 6-phase pipeline execution

# 6. Check results
make session-report
make health
```

## Debugging and Troubleshooting Patterns

### Agent Import Issues (First Line Debugging)
```bash
# Test individual agent imports (most common issue)
PYTHONPATH=/workspace/azure-maintie-rag python -c "from agents.domain_intelligence.agent import domain_intelligence_agent; print('✅ Domain Intelligence')"
PYTHONPATH=/workspace/azure-maintie-rag python -c "from agents.knowledge_extraction.agent import knowledge_extraction_agent; print('✅ Knowledge Extraction')"
PYTHONPATH=/workspace/azure-maintie-rag python -c "from agents.universal_search.agent import universal_search_agent; print('✅ Universal Search')"

# If imports fail, check core dependencies
PYTHONPATH=/workspace/azure-maintie-rag python -c "from agents.core.universal_models import UniversalDomainAnalysis; print('✅ Core models')"
PYTHONPATH=/workspace/azure-maintie-rag python -c "from agents.core.universal_deps import get_universal_deps; print('✅ Dependencies')"
```

### PydanticAI Toolset Verification (Critical Pattern)
```bash
# Verify correct toolset pattern (common cause of Union type errors)
grep -r "@agent\.tool" agents/  # Should return NO results (this pattern causes errors)
grep -r "@.*_toolset\.tool" agents/  # Should find tool registrations (correct pattern)

# Check for correct imports
grep -r "from pydantic_ai.toolsets import FunctionToolset" agents/  # Required import
```

### Azure Service Health Check
```bash
# Service availability diagnostic
PYTHONPATH=/workspace/azure-maintie-rag python -c "
import asyncio
from agents.core.universal_deps import get_universal_deps
async def check():
    deps = await get_universal_deps()
    print('Available services:', list(deps.get_available_services()))
    print('Service status:')
    for service in ['azure_openai', 'cosmos', 'search', 'storage']:
        available = deps.is_service_available(service)
        print(f'  {service}: {\"✅\" if available else \"❌\"}')
asyncio.run(check())
"
```

### Domain Bias Detection (Pre-commit Enforcement)
```bash
# Pre-commit hook validation (enforced on all commits)
./scripts/hooks/pre-commit-domain-bias-check.sh

# Manual domain bias scan
grep -r "legal\|technical\|medical\|financial" agents/ --include="*.py" | grep -v "# OK:" || echo "✅ Clean"

# Check for hardcoded domain categories or thresholds
grep -r "domain.*=.*['\"]" agents/ --include="*.py" | grep -v "discovered\|measured" || echo "✅ Clean"
```

### Version Compatibility Check
```bash
# Critical version verification
python -c "import openai; print(f'OpenAI: {openai.version.VERSION}')"  # Must be 1.98.0
python -c "import pydantic_ai; print(f'PydanticAI: {pydantic_ai.__version__}')"  # Must be 0.6.2+
python -c "import pytest; print(f'Pytest: {pytest.__version__}')"  # Must be 7.4.0+
```

### Configuration Manager Issues
```bash
# Check for deprecated imports (will cause failures)
grep -r "dynamic_config_manager" . --include="*.py" | grep -v "# Deprecated" || echo "✅ Clean"

# Verify correct config manager usage
grep -r "simple_config_manager" agents/ --include="*.py"
```

### Azure Deployment Cleanup Issues (FUNDAMENTAL azd BUG)
```bash
# azd down ALWAYS fails with "no resources found for deployment 'prod-{timestamp}'"
# ROOT CAUSE: azd down is fundamentally broken by design
# 
# What azd down does WRONG:
# 1. Generates NEW deployment name "prod-{current-timestamp}" 
# 2. Searches for that newly generated name (which never existed)
# 3. Fails because it's looking for the wrong deployment
#
# What azd down SHOULD do:
# 1. Find existing deployments in the resource group
# 2. Delete those actual deployments
#
# SOLUTION: Use fixed implementation that works with REAL Azure resources

./scripts/azd-down-fixed.sh --force --purge

# This is NOT a workaround - it's the only way to make resource deletion work
# azd down cannot delete deployments it didn't create in the current session
```

## Azure Deployment Details

### Automated Deployment with Data Pipeline (FIXED)
```bash
make deploy-with-data  # Recommended: Infrastructure + complete data pipeline (FULLY AUTOMATED)
# OR
azd env set AUTO_POPULATE_DATA true && azd up
```

**NEW: Fully automated Option 2 deployment** with smart authentication handling:
1. **Deploys 9 Azure services** with RBAC
2. **Auto-detects authentication** (Container Apps vs local development)  
3. **Checks authentication status** before pipeline execution
4. **Automatically executes postdeploy hook** with Option 2 pipeline
5. **Falls back to post-deployment completion** if auth expires during deployment
6. **Cleans existing Azure data** via Phase 0
7. **Validates all 3 PydanticAI agents** via Phase 1
8. **Uploads REAL documents** & creates embeddings via Phase 2
9. **Runs knowledge extraction** on all docs via Phase 3
10. **Builds tri-modal search** (Vector + Graph + GNN) via Phases 4-6
11. **20-minute timeout protection** prevents hanging
12. **Honest failure reporting** - no fake success when auth issues occur

### Infrastructure Only
```bash
make deploy-infrastructure-only  # Deploy without data
# Then run: make dataflow-full
```

### Environment Sync
```bash
./scripts/deployment/sync-env.sh prod     # Switch to production
./scripts/deployment/sync-env.sh staging  # Switch to staging
make sync-env                              # Sync current environment
```

### Live System Access
```bash
# After deployment completes
./scripts/show-deployment-urls.sh

# Frontend Chat Interface
https://ca-frontend-maintie-rag-prod.<region>.azurecontainerapps.io

# Backend API
https://ca-backend-maintie-rag-prod.<region>.azurecontainerapps.io/health
https://ca-backend-maintie-rag-prod.<region>.azurecontainerapps.io/docs
https://ca-backend-maintie-rag-prod.<region>.azurecontainerapps.io/api/v1/rag
```

## Key Azure Services (9 integrated)

- **Azure OpenAI**: GPT-4o for all agent intelligence
- **Cognitive Search**: Vector search with 1536D embeddings
- **Cosmos DB**: Graph database with Gremlin API
- **Blob Storage**: Document storage
- **Azure ML**: GNN model training/inference (with async bootstrap)
- **Key Vault**: Secrets management
- **Container Apps**: Backend/frontend hosting
- **Application Insights**: Monitoring
- **Managed Identity**: Passwordless authentication

## Technology Stack (Cost-Optimized)

- **Backend**: Python 3.11+, PydanticAI 0.6.2, FastAPI, OpenAI 1.98.0 (critical version)
- **Frontend**: React 19.1.0, TypeScript 5.8.3, Vite 7.0.4
- **Testing**: pytest 7.4.0+ with real Azure services (no mocks), asyncio_mode=auto
- **Code Quality**: Black (88 char), isort, ESLint 9.30.1
- **CI/CD**: GitHub Actions with OIDC, Azure Developer CLI (azd), Bicep IaC
- **AI Models**: GPT-4o-mini (cost-optimized), text-embedding-ada-002 (cost-effective)

## Project Data & Performance

- **Universal corpus**: Adapts to any documents in `data/raw/` directory
- **Processing**: 35-47 seconds per document (varies by content complexity)
- **Confidence**: 88.4% average extraction confidence
- **Success rate**: 100% (all files processed successfully)
- **Production score**: 95/100

## Domain-Specific Context

### **What Makes This System "Universal"**
- **No hardcoded domains**: System discovers content characteristics instead of categorizing
- **Real-time adaptation**: Each document processed with custom parameters based on analysis
- **Universal data intake**: Place ANY documents in `data/raw/` - system adapts automatically
- **Zero-bias enforcement**: Pre-commit hooks prevent domain assumptions from being coded

### **Common Misconceptions to Avoid**
- ❌ **"This is just another RAG system"** - It's a multi-agent system with mandatory tri-modal search
- ❌ **"We can add domain-specific code"** - Violates core philosophy, blocked by pre-commit hooks
- ❌ **"Vector search is enough"** - ALL three modalities (Vector+Graph+GNN) are mandatory
- ❌ **"We can skip agent validation"** - Agent validation is critical path, required before any development

### **Key Design Philosophy Enforcement**
```bash
# This will FAIL pre-commit (correctly):
if domain == "legal":
    complexity = 0.8

# This will PASS (correct approach):
complexity = measure_vocabulary_complexity(content)
parameters = adapt_based_on_measured_properties(complexity)
```

## Critical System Notes

### Architecture Principles
- **NO MOCKS**: All testing uses real Azure services for validation (`pytest.ini:20`)
- **NO DOMAIN BIAS**: System discovers characteristics, doesn't categorize (enforced by `scripts/hooks/pre-commit-domain-bias-check.sh`)
- **FAIL-FAST**: System fails completely rather than using fallback logic (tri-modal search required)
- **UNIVERSAL DESIGN**: Adapts to any content in `data/raw/` without domain-specific code

### Critical Implementation Rules
- **AGENT PATTERN**: Must use `@toolset.tool`, NEVER `@agent.tool` (causes Union type errors)
- **IMPORTS**: Always use `PYTHONPATH=/workspace/azure-maintie-rag` for all Python operations
- **CONFIG MANAGER**: Use `agents.core.simple_config_manager` only (`dynamic_config_manager` is deprecated)
- **DEPENDENCIES**: All Azure services accessed via `agents.core.universal_deps.py` for consistency
- **AUTHENTICATION**: Auto-detects environment (Container Apps vs local development)

### Version Dependencies (Critical)
- **OpenAI**: Must be 1.98.0 (PydanticAI 0.6.2 compatibility requirement)
- **PydanticAI**: 0.4.10+ with Azure support (`requirements.txt:12`) - Note: Version mismatch requires clarification
- **Pytest**: 7.4.0+ with `asyncio_mode=auto` (`pytest.ini:20`)

### Quick Reference: Critical Code Patterns

**✅ CORRECT Agent Tool Registration:**
```python
# 1. Create toolset FIRST
toolset = FunctionToolset()

# 2. Register tools on TOOLSET
@toolset.tool
async def my_tool(ctx: RunContext[UniversalDeps], input: str):
    return await ctx.deps.azure_openai_client.process(input)

# 3. Create agent WITH toolset
agent = Agent[UniversalDeps, OutputModel](
    model=azure_openai_model,
    toolsets=[toolset],  # Pass toolset here
    deps_type=UniversalDeps
)
```

**❌ WRONG Agent Tool Registration (causes Union type errors):**
```python
# DON'T DO THIS - causes PydanticAI 0.6.2 Union type errors
@agent.tool  # ❌ This pattern breaks
async def my_tool(ctx: RunContext[UniversalDeps], input: str):
    return result
```

**✅ CORRECT Async Function Execution:**
```python
# Always use proper async patterns with Azure services
async def process_with_azure():
    deps = await get_universal_deps()
    result = await some_azure_operation(deps)
    return result

# For scripts: always use asyncio.run()
if __name__ == "__main__":
    result = asyncio.run(process_with_azure())
```

**✅ CORRECT Environment Variable Pattern:**
```bash
# Always set these for ANY Python script execution
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/workspace/azure-maintie-rag python script.py
```

### Cost Optimization Features
- **SCALE-TO-ZERO**: Container apps scale to zero when not in use
- **FREE TIERS**: Cosmos DB (1M RU/s + 25GB/month), Cognitive Search (50MB, 3 indexes, 10K docs)
- **CPU-ONLY AZURE ML**: Cost-optimized CPU training ($0.168/hour, auto-shutdown enabled)
- **GPT-4o-MINI**: Cost-effective model replacement

### Session & Logging
- **CLEAN SESSIONS**: Makefile replaces logs, doesn't accumulate (`Makefile:14-60`)
- **SESSION TRACKING**: Enterprise session management with real-time Azure service monitoring
- **CUMULATIVE REPORTING**: Long-term session history preserved in `logs/cumulative_dataflow_report.md`