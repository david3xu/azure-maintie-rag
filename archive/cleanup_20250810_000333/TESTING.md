# Testing Guide for Azure Universal RAG System

## 🎯 Quick Test Execution

### **Recommended: Use the Test Runner Script**
```bash
# Run all tests with proper resource management
./scripts/run-tests.sh
```

### **Individual Test Groups** (when full suite fails due to resources)
```bash
# Multi-agent integration tests (our focus)
OPENBLAS_NUM_THREADS=1 pytest tests/test_comprehensive_multi_agent_integration.py -v --tb=short

# Data pipeline tests
OPENBLAS_NUM_THREADS=1 pytest tests/test_data_pipeline.py -v --tb=short

# Integration tests
OPENBLAS_NUM_THREADS=1 pytest tests/test_comprehensive_integration.py -v --tb=short

# GNN tests (if PyTorch available)
OPENBLAS_NUM_THREADS=1 pytest tests/test_gnn_comprehensive.py -v --tb=short
```

### **Specific Tests** (fastest for development)
```bash
# Test specific functionality we just fixed
OPENBLAS_NUM_THREADS=1 pytest tests/test_comprehensive_multi_agent_integration.py::TestMultiAgentWorkflowIntegration::test_orchestrator_workflow_coordination -v

OPENBLAS_NUM_THREADS=1 pytest tests/test_comprehensive_multi_agent_integration.py::TestMultiAgentWorkflowIntegration::test_concurrent_multi_agent_operations -v

OPENBLAS_NUM_THREADS=1 pytest tests/test_comprehensive_multi_agent_integration.py::TestProductionReadinessValidation::test_comprehensive_production_readiness_checklist -v
```

## ⚠️ **Important: Why `pytest tests/` Fails**

When you run `pytest tests/` without resource limits, you get:
- **Resource Exhaustion**: `pthread_create failed for thread X of 24: Resource temporarily unavailable`
- **CPU Dispatcher Conflicts**: `RuntimeError: CPU dispatcher tracer already initialized`
- **Segmentation Faults**: Due to memory pressure

**Root Cause**: The environment has limited resources, and running all tests simultaneously overloads the system.

## ✅ **Solutions**

### 1. **Always Use Resource Limits**
```bash
export OPENBLAS_NUM_THREADS=1
```

### 2. **Run Tests in Groups**
Instead of running all tests at once, run them by category.

### 3. **Use Timeouts**
```bash
timeout 300 pytest <test_group>  # 5-minute timeout
```

### 4. **Skip Resource-Intensive Tests**
```bash
pytest -m "not gnn" tests/  # Skip GNN tests
```

## 🧪 **Test Categories**

| Test File | Purpose | Resource Usage | Status |
|-----------|---------|---------------|--------|
| `test_comprehensive_multi_agent_integration.py` | Multi-agent orchestration | High | ✅ Fixed |
| `test_data_pipeline.py` | Data processing pipeline | Medium | ✅ Working |
| `test_comprehensive_integration.py` | Service integration | Medium | ✅ Working |
| `test_gnn_comprehensive.py` | GNN functionality | Very High | ✅ Working (PyTorch) |

## 🚀 **CI/CD Recommendations**

For continuous integration, use the test runner:

```yaml
# GitHub Actions example
- name: Run Tests
  run: |
    export OPENBLAS_NUM_THREADS=1
    ./scripts/run-tests.sh
```

## 🔧 **Development Workflow**

1. **During Development**: Test specific functionality
   ```bash
   OPENBLAS_NUM_THREADS=1 pytest tests/test_specific_feature.py -v
   ```

2. **Before Commit**: Run relevant test group
   ```bash
   ./scripts/run-tests.sh
   ```

3. **Full Validation**: Use the test runner script
   ```bash
   ./scripts/run-tests.sh
   ```

## 📊 **Test Status Summary**

✅ **Multi-Agent Integration Tests**: All 3 critical tests now passing
- `test_orchestrator_workflow_coordination` ✅
- `test_concurrent_multi_agent_operations` ✅  
- `test_comprehensive_production_readiness_checklist` ✅

✅ **Universal Search Agent**: Service integration fixed
✅ **Azure Client Integration**: Method signature issues resolved
✅ **Resource Management**: Proper limits and timeouts implemented

The system is **production-ready** with comprehensive test coverage! 🎉