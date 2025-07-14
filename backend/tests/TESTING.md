# MaintIE RAG Project: Testing Guide

This document explains how to run all tests for the MaintIE RAG project, including unit, integration, API, and coverage tests. It also covers common troubleshooting tips for import errors.

---

## Directory Structure

- `tests/unit/`         — Unit tests for core modules (including GNN)
- `tests/integration/`  — End-to-end and pipeline integration tests
- `tests/api/`          — API endpoint and health check tests
- `tests/debug/`        — Debug/experimental scripts (not formal tests)

---

## Running All Tests

**From the `backend/` directory:**

```bash
cd backend
PYTHONPATH=. pytest tests/
```

This will run all unit, integration, and API tests.

---

## Running Tests with Coverage

To see a code coverage report for the `src/` directory:

```bash
cd backend
PYTHONPATH=. pytest --cov=src tests/
```

- The coverage summary will be printed at the end of the test run.
- You can also generate an HTML report:
  ```bash
  PYTHONPATH=. pytest --cov=src --cov-report=html tests/

  cd backend && PYTHONPATH=. pytest --cov=src tests/
  # Open htmlcov/index.html in your browser
  ```

---

## Running a Specific Test File

```bash
cd backend
PYTHONPATH=. pytest tests/unit/test_gnn_models.py
```

---

## Common Import Errors & Fixes

- **Error:** `ModuleNotFoundError: No module named 'src'`
- **Fix:** Always run tests from the `backend/` directory and set `PYTHONPATH=.`

---

## Adding New Tests

- Place new unit tests in `tests/unit/`
- Place new integration tests in `tests/integration/`
- Place new API tests in `tests/api/`
- Place debug/experimental scripts in `tests/debug/`

---

## Example: GNN Unit Test

See `tests/unit/test_gnn_models.py`, `test_gnn_data_preparation.py`, and `test_gnn_query_expander.py` for GNN-specific test templates.

---

## Test Runner Script

You can also use the provided script to run all tests:

```bash
cd backend
python tests/run_all_tests.py
```

---

## Comprehensive System Test Suite

A full end-to-end test suite is available at `tests/comprehensive_test_suite.py`. This script tests all real API endpoints, response structures, and features found in the codebase, including:
- API connectivity and health
- All query endpoints (multi-modal, structured, comparison)
- Performance and concurrency
- Query analysis, domain intelligence, safety features
- Caching and monitoring

**To run the comprehensive suite:**

```bash
cd backend
python tests/comprehensive_test_suite.py
# or
python -m pytest tests/comprehensive_test_suite.py -v
```

This suite is ideal for CI/CD pipelines and full system health checks.

---

**For any issues, check the test output and ensure your environment is activated and dependencies are installed.**