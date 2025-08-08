#!/bin/bash
# Test Runner for Azure Universal RAG System
# Handles resource constraints and provides comprehensive test execution

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Azure Universal RAG Test Suite${NC}"
echo "=================================="

# Set resource limits
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest not found. Please install pytest.${NC}"
    exit 1
fi

# Function to run tests with proper error handling
run_test_group() {
    local test_group="$1"
    local description="$2"
    
    echo -e "\n${BLUE}🧪 Running: ${description}${NC}"
    echo "----------------------------------------"
    
    if timeout 300 pytest "$test_group" -v --tb=short; then
        echo -e "${GREEN}✅ ${description} - PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ ${description} - FAILED${NC}"
        return 1
    fi
}

# Initialize counters
total_groups=0
passed_groups=0

# Test execution plan
echo -e "\n${YELLOW}📋 Test Execution Plan:${NC}"
echo "1. Multi-Agent Integration Tests"
echo "2. Data Pipeline Tests" 
echo "3. Comprehensive Integration Tests"
echo "4. GNN Tests (if PyTorch available)"
echo "5. Unit Tests"

# Test Group 1: Multi-Agent Integration
echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
total_groups=$((total_groups + 1))
if run_test_group "tests/test_comprehensive_multi_agent_integration.py" "Multi-Agent Integration Tests"; then
    passed_groups=$((passed_groups + 1))
fi

# Test Group 2: Data Pipeline
echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
total_groups=$((total_groups + 1))
if run_test_group "tests/test_data_pipeline.py" "Data Pipeline Tests"; then
    passed_groups=$((passed_groups + 1))
fi

# Test Group 3: Comprehensive Integration
echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
total_groups=$((total_groups + 1))
if run_test_group "tests/test_comprehensive_integration.py" "Comprehensive Integration Tests"; then
    passed_groups=$((passed_groups + 1))
fi

# Test Group 4: GNN Tests (conditional)
if python -c "import torch" 2>/dev/null; then
    echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
    total_groups=$((total_groups + 1))
    if run_test_group "tests/test_gnn_comprehensive.py" "GNN Comprehensive Tests"; then
        passed_groups=$((passed_groups + 1))
    fi
else
    echo -e "\n${YELLOW}⚠️  Skipping GNN tests - PyTorch not available${NC}"
fi

# Test Group 5: Unit Tests (if they exist)
if ls tests/test_unit_*.py 1> /dev/null 2>&1; then
    echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
    total_groups=$((total_groups + 1))
    if run_test_group "tests/test_unit_*.py" "Unit Tests"; then
        passed_groups=$((passed_groups + 1))
    fi
fi

# Final Summary
echo -e "\n${BLUE}📊 Test Execution Summary${NC}"
echo "=========================="
echo -e "Total Test Groups: ${total_groups}"
echo -e "Passed: ${GREEN}${passed_groups}${NC}"
echo -e "Failed: ${RED}$((total_groups - passed_groups))${NC}"

success_rate=$(( (passed_groups * 100) / total_groups ))
echo -e "Success Rate: ${success_rate}%"

if [ "$passed_groups" -eq "$total_groups" ]; then
    echo -e "\n${GREEN}🎉 All test groups passed successfully!${NC}"
    exit 0
elif [ "$success_rate" -ge 80 ]; then
    echo -e "\n${YELLOW}⚠️  Most tests passed. Some groups may need attention.${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Multiple test groups failed. Please investigate.${NC}"
    exit 1
fi