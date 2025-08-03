#!/usr/bin/env python3
"""
Config Structure Refactoring Script

Refactors the config directory to have clear, intuitive naming and structure:

Current messy structure:
├── consolidated_config.py
├── data_driven_schema.py  
├── settings.py
├── timeout_config.py
├── unified_data_driven_config.yaml
├── v2_config_models.py

New clean structure:
├── README.md                     # How the config system works
├── main.py                       # Main config manager (was consolidated_config.py)
├── models.py                     # Data models/schema (was data_driven_schema.py)
├── azure_settings.py             # Azure environment settings (was settings.py)
├── timeouts.py                   # Timeout configurations (was timeout_config.py)
├── config.yaml                   # Main generated config (was unified_data_driven_config.yaml)
├── legacy_models.py              # Legacy V2 models (was v2_config_models.py)
├── generated/                    # Generated configurations
│   └── domains/
│       └── azure_cloud.yaml     # Domain configs (renamed)
└── environments/                 # Environment-specific files
    ├── development.env
    └── staging.env
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List


class ConfigStructureRefactor:
    """Refactors config directory for clarity and intuitive naming"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        
        # Mapping of old names to new names
        self.file_renames = {
            "consolidated_config.py": "main.py",
            "data_driven_schema.py": "models.py", 
            "settings.py": "azure_settings.py",
            "timeout_config.py": "timeouts.py",
            "unified_data_driven_config.yaml": "config.yaml",
            "v2_config_models.py": "legacy_models.py"
        }
        
        # Directory restructuring
        self.dir_renames = {
            "domains": "generated/domains"
        }
        
        # Import mapping for code updates
        self.import_updates = {
            "config.consolidated_config": "config.main",
            "config.data_driven_schema": "config.models",
            "config.settings": "config.azure_settings", 
            "config.timeout_config": "config.timeouts",
            "config.v2_config_models": "config.legacy_models"
        }
        
        self.refactor_report = {
            "files_renamed": [],
            "directories_restructured": [],
            "imports_updated": [],
            "files_created": []
        }
    
    def create_readme(self):
        """Create comprehensive README for config system"""
        readme_content = '''# Azure RAG Configuration System

## Overview

This directory contains the **unified, data-driven configuration system** for the Azure Universal RAG project. The system generates all configurations automatically from raw data analysis.

## 🎯 Key Principles

1. **Data-Driven**: All domain configurations generated from raw text analysis
2. **Schema-Validated**: Pydantic models ensure consistency
3. **Single Source of Truth**: One unified configuration system
4. **Environment-Aware**: Supports dev/staging/prod environments
5. **Reproducible**: Complete pipeline from raw data to config

## 📁 Directory Structure

```
config/
├── README.md                     # This documentation
├── main.py                       # 🎯 Main configuration manager
├── models.py                     # 🎯 Data models and schemas  
├── azure_settings.py             # Azure environment settings
├── timeouts.py                   # System timeout configurations
├── config.yaml                   # 🎯 Main generated configuration
├── legacy_models.py              # Legacy V2 models (future integration)
├── generated/                    # 🎯 Generated configurations
│   └── domains/
│       └── azure_cloud.yaml     # Generated domain configurations
└── environments/                 # Environment-specific settings
    ├── development.env
    └── staging.env
```

## 🚀 Quick Start

### 1. Get Configuration Manager

```python
from config.main import get_config_manager

# Get the unified configuration manager
config = get_config_manager()

# List available domains
domains = config.list_domains()
print(f"Available domains: {domains}")
```

### 2. Access Domain Configuration

```python
# Get domain-specific configuration
domain_config = config.get_domain_config("azure_cloud")

# Access domain properties
entities = domain_config.primary_entities
concepts = domain_config.key_concepts
vocabulary = domain_config.technical_vocabulary
```

### 3. Access Azure Service Configuration

```python
# Get Azure service configurations
openai_config = config.get_azure_service_config("openai")
search_config = config.get_azure_service_config("search")
cosmos_config = config.get_azure_service_config("cosmos")
```

### 4. Validate Configuration

```python
from config.main import validate_all_config

# Validate entire configuration system
validation = validate_all_config()

if validation["valid"]:
    print("✅ Configuration is valid")
else:
    print("⚠️ Configuration issues found:")
    for error in validation["errors"]:
        print(f"  - {error}")
```

## 🔄 Data-Driven Pipeline

The configuration system is generated through this pipeline:

1. **Raw Data Analysis** (`data/raw/*.md`) → Extract entities, relationships, concepts
2. **Schema Validation** (`models.py`) → Validate quality and structure  
3. **Domain Generation** (`generated/domains/`) → Create domain-specific configs
4. **Unified Config** (`config.yaml`) → Merge into single configuration
5. **Validation** → Ensure prompt flow readiness

### Regenerate from Raw Data

```bash
# Regenerate entire configuration from raw data
python scripts/validate_knowledge_quality.py
```

## 📋 Configuration Components

### Core Files

- **`main.py`** - Main configuration manager and API
- **`models.py`** - Pydantic schemas for data-driven configuration
- **`config.yaml`** - Main generated configuration file

### Supporting Files

- **`azure_settings.py`** - Azure environment variables and settings
- **`timeouts.py`** - System timeout configurations
- **`legacy_models.py`** - Legacy V2 models for future integration

### Generated Files

- **`generated/domains/`** - Domain-specific configurations generated from data
- **`environments/`** - Environment-specific settings (dev/staging/prod)

## 🎯 Configuration Schema

### Domain Configuration

Each domain configuration includes:

```yaml
domain_name: programming_language_config  # From Programming-Language/ directory
data_quality: excellent
entity_count: 0
entity_types:
  azure_service: [Azure Machine Learning, Azure ML, ...]
  ml_concept: [MLOps, Machine Learning, ...]
key_concepts: [Machine Learning, Data Science, ...]
technical_vocabulary: [REST API, Python SDK, ...]
relationship_patterns: [Azure ML -> provides -> MLOps capabilities]
generated_from_data: true
# NOTE: No domain_type field - domains are pure directory names
```

### Unified Configuration

The main configuration includes:

```yaml
domain_configs: {azure_cloud: {...}}
tri_modal_search: {enabled_modalities: [vector, graph, gnn]}
competitive_advantage: {confidence_threshold: 0.7}
azure_services: {openai: {...}, search: {...}}
```

## 🔧 Environment Configuration

### Development

```bash
# Load development environment
export ENVIRONMENT=development
```

### Production

```bash
# Load production environment  
export ENVIRONMENT=production
```

### Custom Environment

```bash
# Use custom environment settings
export ENVIRONMENT=custom
```

## 🏗️ Architecture

### Layers

1. **Data Layer** - Raw text files → Extracted knowledge
2. **Schema Layer** - Pydantic models → Validated structures
3. **Generation Layer** - Knowledge → Domain configurations  
4. **Unified Layer** - Domain configs → Single configuration
5. **Application Layer** - Configuration → Service configs

### Data Flow

```
Raw Data → Knowledge Extraction → Schema Validation → Domain Generation → Unified Config → Application Use
```

## 🧪 Testing

### Validate Configuration

```python
from config.main import validate_all_config

validation = validate_all_config()
assert validation["valid"], "Configuration must be valid"
assert validation["details"]["data_driven"], "Must use data-driven config"
```

### Test Domain Access

```python
from config.main import get_domain_config

domain = get_domain_config("azure_cloud")
assert domain is not None, "Domain must exist"
assert len(domain.primary_entities) > 0, "Must have entities"
```

## 🚨 Troubleshooting

### Configuration Not Loading

1. Check that `config.yaml` exists and is valid YAML
2. Verify Pydantic models in `models.py` are correct
3. Run validation: `python -c "from config.main import validate_all_config; print(validate_all_config())"`

### Domain Configurations Missing

1. Run data pipeline: `python scripts/validate_knowledge_quality.py`
2. Check `generated/domains/` directory for domain files
3. Verify raw data exists in `data/raw/`

### Azure Service Configuration Issues

1. Check environment variables: `cat .env`
2. Verify Azure settings: `python -c "from config.azure_settings import Settings; print(Settings().model_dump())"`
3. Test connectivity: `python scripts/test_azure_connectivity.py`

## 📚 Related Documentation

- **[Data Pipeline](../scripts/validate_knowledge_quality.py)** - Raw data to config pipeline
- **[Azure Setup](../docs/getting-started/QUICK_START.md)** - Azure service configuration
- **[Architecture](../docs/architecture/SYSTEM_ARCHITECTURE.md)** - System overview

## 🔄 Migration Notes

This unified configuration system replaces:
- ❌ `config_loader.py` → `main.py`
- ❌ `azure_config_validator.py` → `main.py` validation
- ❌ `production_config.py` → `timeouts.py`
- ❌ `inter_layer_contracts.py` → `models.py`

All legacy imports have been automatically updated.

---

**Generated by**: Data-driven configuration system  
**Last Updated**: 2025-08-02  
**Version**: 1.0.0
'''
        
        readme_path = self.config_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.refactor_report["files_created"].append("README.md")
        print(f"✅ Created comprehensive README.md")
    
    def rename_files(self):
        """Rename files with clear, intuitive names"""
        print("📝 Renaming files with clear names...")
        
        for old_name, new_name in self.file_renames.items():
            old_path = self.config_dir / old_name
            new_path = self.config_dir / new_name
            
            if old_path.exists():
                shutil.move(str(old_path), str(new_path))
                print(f"   📄 Renamed: {old_name} → {new_name}")
                self.refactor_report["files_renamed"].append(f"{old_name} → {new_name}")
            else:
                print(f"   ⚠️ File not found: {old_name}")
    
    def restructure_directories(self):
        """Restructure directories for clarity"""
        print("📁 Restructuring directories...")
        
        # Create generated directory
        generated_dir = self.config_dir / "generated"
        generated_dir.mkdir(exist_ok=True)
        
        # Move domains to generated/domains
        old_domains_dir = self.config_dir / "domains"
        new_domains_dir = generated_dir / "domains"
        
        if old_domains_dir.exists():
            shutil.move(str(old_domains_dir), str(new_domains_dir))
            print(f"   📁 Moved: domains/ → generated/domains/")
            self.refactor_report["directories_restructured"].append("domains/ → generated/domains/")
        
        # Rename domain files to be cleaner
        if new_domains_dir.exists():
            for domain_file in new_domains_dir.glob("*.yaml"):
                if "azure_cloud_config" in domain_file.name:
                    new_domain_file = new_domains_dir / "azure_cloud.yaml"
                    shutil.move(str(domain_file), str(new_domain_file))
                    print(f"   📄 Renamed: {domain_file.name} → azure_cloud.yaml")
    
    def update_import_references(self):
        """Update import references throughout codebase"""
        print("🔄 Updating import references...")
        
        # Files that might need updates
        files_to_check = [
            "services/query_service.py",
            "services/workflow_service.py", 
            "services/agent_service.py",
            "tests/validation/validate_layer_boundaries.py",
            "scripts/validate_knowledge_quality.py",
            "scripts/complete_config_cleanup.py"
        ]
        
        updated_files = []
        
        for file_path in files_to_check:
            full_path = Path(file_path)
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply import updates
                    for old_import, new_import in self.import_updates.items():
                        if old_import in content:
                            content = content.replace(old_import, new_import)
                            print(f"   🔄 Updated import in {file_path}: {old_import} → {new_import}")
                    
                    # Only write if changed
                    if content != original_content:
                        with open(full_path, 'w') as f:
                            f.write(content)
                        updated_files.append(file_path)
                        self.refactor_report["imports_updated"].append(file_path)
                
                except Exception as e:
                    print(f"   ⚠️ Could not update {file_path}: {e}")
        
        if not updated_files:
            print("   ✅ No import references needed updating")
    
    def create_init_file(self):
        """Create clean __init__.py for package"""
        init_content = '''"""
Azure Universal RAG Configuration System

This package provides a unified, data-driven configuration system that generates
all configurations automatically from raw data analysis.

Quick Start:
    from config.main import get_config_manager
    
    config = get_config_manager()
    domains = config.list_domains()
"""

# Main API exports
from .main import (
    get_config_manager,
    get_domain_config, 
    get_azure_config,
    validate_all_config
)

# Core models
from .models import (
    DataDrivenExtraction,
    DomainConfiguration, 
    UnifiedDataDrivenConfig
)

# Settings
from .azure_settings import Settings

__version__ = "1.0.0"
__all__ = [
    "get_config_manager",
    "get_domain_config",
    "get_azure_config", 
    "validate_all_config",
    "DataDrivenExtraction",
    "DomainConfiguration",
    "UnifiedDataDrivenConfig",
    "Settings"
]
'''
        
        init_path = self.config_dir / "__init__.py"
        with open(init_path, 'w') as f:
            f.write(init_content)
        
        print(f"✅ Updated __init__.py with clean exports")
    
    def validate_refactored_structure(self):
        """Validate the refactored configuration structure"""
        print("✅ Validating refactored structure...")
        
        try:
            # Test imports with new names
            import sys
            sys.path.append('.')
            
            from config.main import get_config_manager, validate_all_config
            
            config_manager = get_config_manager()
            validation = validate_all_config()
            
            print(f"   ✅ New imports working")
            print(f"   📊 Config manager: {len(config_manager.list_domains())} domains")
            print(f"   🔍 Data-driven: {validation['details']['data_driven']}")
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
    
    def generate_refactor_summary(self):
        """Generate refactoring summary"""
        print("\n📊 REFACTORING SUMMARY")
        print("=" * 50)
        
        print(f"Files renamed: {len(self.refactor_report['files_renamed'])}")
        for rename in self.refactor_report['files_renamed']:
            print(f"   📄 {rename}")
        
        print(f"\nDirectories restructured: {len(self.refactor_report['directories_restructured'])}")
        for restructure in self.refactor_report['directories_restructured']:
            print(f"   📁 {restructure}")
        
        print(f"\nFiles created: {len(self.refactor_report['files_created'])}")
        for created in self.refactor_report['files_created']:
            print(f"   ✨ {created}")
        
        print(f"\nImports updated: {len(self.refactor_report['imports_updated'])}")
        for updated in self.refactor_report['imports_updated']:
            print(f"   🔄 {updated}")
        
        print(f"\n🎯 NEW CLEAN STRUCTURE:")
        print(f"config/")
        print(f"├── README.md              # 📚 Complete documentation")
        print(f"├── main.py               # 🎯 Main config manager")
        print(f"├── models.py             # 🎯 Data schemas")
        print(f"├── config.yaml           # 🎯 Generated config")
        print(f"├── azure_settings.py     # ⚙️ Azure settings")
        print(f"├── timeouts.py           # ⏱️ Timeout configs")
        print(f"├── generated/domains/    # 🎯 Generated domains")
        print(f"└── environments/         # 🌍 Environment files")
    
    def run_refactor(self):
        """Run the complete refactoring"""
        print("🔧 CONFIGURATION STRUCTURE REFACTORING")
        print("=" * 60)
        print("Renaming files and restructuring for clarity")
        print("=" * 60)
        
        # Execute refactoring steps
        self.create_readme()
        self.rename_files()
        self.restructure_directories()
        self.create_init_file()
        self.update_import_references()
        self.validate_refactored_structure()
        self.generate_refactor_summary()
        
        print("\n🎉 REFACTORING COMPLETED!")
        print("Configuration directory now has clear, intuitive structure!")
        
        return self.refactor_report


def main():
    """Run configuration structure refactoring"""
    refactor = ConfigStructureRefactor()
    return refactor.run_refactor()


if __name__ == "__main__":
    result = main()
    print(f"\n🏁 Refactor Result: {len(result['files_renamed'])} files renamed")