# Debug Directory

This directory contains development and deployment debugging scripts for the MaintIE Enhanced RAG system.

## 📁 Directory Structure

```
backend/debug/                    # Development & deployment debugging
├── debug_pipeline.py            # Pipeline debugging (QueryType enum issues)
├── debug_query_type.py          # Query type analysis debugging
├── deploy_gnn_integration.py    # GNN integration deployment & testing
└── README.md                    # This file

backend/tests/debug/             # Test-specific debugging
├── debug_data_structure.py      # Data structure validation for tests
└── debug_entity_extraction.py   # Entity extraction testing
```

## 🛠️ Debug Scripts

### Development & Deployment Scripts

#### `debug_pipeline.py`
**Purpose**: Debug pipeline issues, specifically QueryType enum serialization problems
**Usage**:
```bash
cd backend
python debug/debug_pipeline.py
```
**Features**:
- Tests full pipeline execution
- Identifies where QueryType enum becomes string
- Tests serialization/deserialization
- Provides detailed error tracing

#### `debug_query_type.py`
**Purpose**: Debug query type analysis and classification
**Usage**:
```bash
cd backend
python debug/debug_query_type.py
```
**Features**:
- Tests query type detection
- Validates enum handling
- Debugs classification logic

#### `deploy_gnn_integration.py`
**Purpose**: Deploy and test GNN integration components
**Usage**:
```bash
cd backend
python debug/deploy_gnn_integration.py
```
**Features**:
- Tests GNN data preparation
- Validates GNN query expansion
- Tests complete GNN pipeline
- Deployment verification

### Test-Specific Scripts

#### `debug_data_structure.py` (in `tests/debug/`)
**Purpose**: Inspect MaintIE data structure for testing
**Usage**:
```bash
cd backend
python tests/debug/debug_data_structure.py
```
**Features**:
- Inspects JSON data structure
- Validates data format for tests
- Sample data analysis

#### `debug_entity_extraction.py` (in `tests/debug/`)
**Purpose**: Debug entity extraction process for testing
**Usage**:
```bash
cd backend
python tests/debug/debug_entity_extraction.py
```
**Features**:
- Tests entity extraction step-by-step
- Validates entity creation
- Debugs extraction pipeline

## 🚀 Quick Debug Commands

### Pipeline Debugging
```bash
# Debug pipeline issues
cd backend && python debug/debug_pipeline.py

# Debug query type analysis
cd backend && python debug/debug_query_type.py
```

### GNN Integration Testing
```bash
# Test GNN integration
cd backend && python debug/deploy_gnn_integration.py
```

### Test-Specific Debugging
```bash
# Debug data structure for tests
cd backend && python tests/debug/debug_data_structure.py

# Debug entity extraction for tests
cd backend && python tests/debug/debug_entity_extraction.py
```

## 🔧 Debug Workflow

### 1. Development Debugging
When working on new features or fixing bugs:
1. Use scripts in `backend/debug/`
2. Run with proper path setup
3. Check for pipeline issues
4. Validate GNN integration

### 2. Test Debugging
When debugging test failures:
1. Use scripts in `backend/tests/debug/`
2. Focus on test-specific issues
3. Validate data structures
4. Check entity extraction

### 3. Deployment Verification
Before deploying:
1. Run `deploy_gnn_integration.py`
2. Verify all components work
3. Check integration points
4. Validate pipeline flow

## 📋 Best Practices

### Running Debug Scripts
- Always run from the `backend/` directory
- Check Python path setup
- Verify dependencies are installed
- Review output carefully

### Debugging Tips
- Start with data structure validation
- Test components individually
- Use step-by-step debugging
- Check for serialization issues

### Maintenance
- Keep debug scripts updated
- Remove obsolete debug code
- Document new debug scenarios
- Test debug scripts regularly

## 🔄 Integration with Makefile

You can add debug commands to your Makefile:

```makefile
# Add to backend/Makefile
debug-pipeline:
	@echo "🔍 Debugging pipeline..."
	python debug/debug_pipeline.py

debug-gnn:
	@echo "🧠 Testing GNN integration..."
	python debug/deploy_gnn_integration.py

debug-tests:
	@echo "🧪 Debugging test data..."
	python tests/debug/debug_data_structure.py
```

---

**Last Updated**: 2024
**Maintainer**: Development Team
**Version**: 1.0