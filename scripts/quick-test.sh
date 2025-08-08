#!/bin/bash
# Quick Test Validation - Tests our specific fixes
# Use this to validate the multi-agent integration fixes quickly

set -e
export OPENBLAS_NUM_THREADS=1

echo "🧪 Quick Test Validation - Multi-Agent Integration Fixes"
echo "========================================================"

# Test the specific tests we fixed
echo ""
echo "1️⃣ Testing Orchestrator Workflow Coordination..."
if timeout 60 pytest tests/test_comprehensive_multi_agent_integration.py::TestMultiAgentWorkflowIntegration::test_orchestrator_workflow_coordination -v --tb=short; then
    echo "   ✅ PASSED"
else
    echo "   ❌ FAILED"
fi

echo ""
echo "2️⃣ Testing Concurrent Multi-Agent Operations..."  
if timeout 120 pytest tests/test_comprehensive_multi_agent_integration.py::TestMultiAgentWorkflowIntegration::test_concurrent_multi_agent_operations -v --tb=short; then
    echo "   ✅ PASSED"
else
    echo "   ❌ FAILED"
fi

echo ""
echo "3️⃣ Testing Production Readiness Checklist..."
if timeout 90 pytest tests/test_comprehensive_multi_agent_integration.py::TestProductionReadinessValidation::test_comprehensive_production_readiness_checklist -v --tb=short; then
    echo "   ✅ PASSED"
else
    echo "   ❌ FAILED"
fi

echo ""
echo "🎉 Quick validation complete!"
echo "For full test suite, use: ./scripts/run-tests.sh"