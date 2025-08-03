# Config Layer Boundary Fix Plan

**Issue Identified**: Config directory violates layer boundaries by mixing Infrastructure, Services, and Agent responsibilities.

## Current Violations

```
config/ (MIXED RESPONSIBILITIES - WRONG!)
├── settings.py              # ✅ Infrastructure Layer 
├── azure_settings.py        # ✅ Infrastructure Layer
├── models.py               # ❌ Should be Services Layer
├── extraction_interface.py # ❌ Should be Services Layer  
├── generated/domains/       # ❌ Should be Agent 1 domain
└── agents/                 # ❌ Should be Agent domain
```

## Correct Layer Architecture

```
API Layer → Services Layer → Infrastructure Layer → Azure Services
```

### Infrastructure Layer (config/)
**Purpose**: Azure connection settings only
```
config/
├── settings.py          # Azure endpoints, credentials
├── azure_settings.py    # Azure service configuration
├── environments/        # Environment variables
└── timeouts.py         # Infrastructure timeouts
```

### Services Layer (services/)  
**Purpose**: Business logic models and interfaces
```
services/
├── models/
│   ├── extraction_models.py    # Extraction business models
│   ├── domain_models.py        # Domain business models
│   └── query_models.py         # Query business models
└── interfaces/
    ├── extraction_interface.py # Business logic interfaces
    └── domain_interface.py     # Domain service interfaces
```

### Agent Layer (agents/)
**Purpose**: Agent 1 generated configurations and agent-specific models
```
agents/domain_intelligence/
├── generated_configs/       # Agent 1 learned configurations
│   ├── programming_language_config.yaml
│   └── medical_config.yaml  
├── models.py               # Agent 1's self-contained models
└── config_generator.py     # Agent 1's configuration generation logic
```

## Migration Steps

### Step 1: Move Business Logic Models to Services Layer
- Move `config/models.py` → `services/models/domain_models.py`
- Move `config/extraction_interface.py` → `services/interfaces/extraction_interface.py`

### Step 2: Move Agent 1 Generated Configs to Agent Domain  
- Move `config/generated/domains/` → `agents/domain_intelligence/generated_configs/`
- Move `config/agents/` → `agents/domain_intelligence/agent_configs/`

### Step 3: Clean Infrastructure Layer
- Keep only Azure settings in `config/`
- Remove business logic from infrastructure layer

### Step 4: Update Import Paths
- Update all imports to respect new layer boundaries
- Ensure API → Services → Infrastructure → Azure Services flow

## Benefits

1. **Clear Layer Separation**: Each layer has single responsibility
2. **Agent 1 Self-Contained**: All Agent 1 configs in agent domain
3. **Proper Dependency Flow**: API → Services → Infrastructure → Azure
4. **Maintainability**: Clear boundaries prevent architectural drift

## Architecture Compliance

✅ **Agent 1 Principle**: "ALL hardcoded values MUST be contained ONLY in Agent 1"
✅ **Layer Boundaries**: Clear separation of concerns
✅ **PydanticAI Compliance**: Agent-centric configuration management