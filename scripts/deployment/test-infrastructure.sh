#!/bin/bash
# Infrastructure Testing Script
# Validates Azure Universal RAG infrastructure deployment

set -e

echo "🧪 Testing Azure Universal RAG Infrastructure..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test functions
test_azd_setup() {
    echo -e "${BLUE}🔍 Testing azd setup...${NC}"

    # Check azd installation
    if ! command -v azd &> /dev/null; then
        echo -e "${RED}❌ Azure Developer CLI (azd) not installed${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ azd is installed${NC}"

    # Check if logged in
    if ! azd auth show &> /dev/null; then
        echo -e "${YELLOW}⚠️  Not logged in to Azure${NC}"
        echo "Run: azd auth login"
        return 1
    fi
    echo -e "${GREEN}✅ Authenticated with Azure${NC}"

    return 0
}

test_bicep_syntax() {
    echo -e "${BLUE}🔍 Testing Bicep syntax...${NC}"

    # Check if bicep is available
    if ! command -v bicep &> /dev/null; then
        echo -e "${YELLOW}⚠️  Bicep CLI not found, trying az bicep...${NC}"
        if ! az bicep version &> /dev/null; then
            echo -e "${RED}❌ Bicep not available${NC}"
            return 1
        fi
        BICEP_CMD="az bicep"
    else
        BICEP_CMD="bicep"
    fi

    # Test main.bicep syntax
    echo "Testing main.bicep..."
    if $BICEP_CMD build infra/main.bicep --stdout > /dev/null; then
        echo -e "${GREEN}✅ main.bicep syntax valid${NC}"
    else
        echo -e "${RED}❌ main.bicep syntax error${NC}"
        return 1
    fi

    # Test module syntax
    for module in infra/modules/*.bicep; do
        if [ -f "$module" ]; then
            echo "Testing $(basename $module)..."
            if $BICEP_CMD build "$module" --stdout > /dev/null; then
                echo -e "${GREEN}✅ $(basename $module) syntax valid${NC}"
            else
                echo -e "${RED}❌ $(basename $module) syntax error${NC}"
                return 1
            fi
        fi
    done

    return 0
}

test_azd_config() {
    echo -e "${BLUE}🔍 Testing azd configuration...${NC}"

    # Check azure.yaml exists
    if [ ! -f "azure.yaml" ]; then
        echo -e "${RED}❌ azure.yaml not found${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ azure.yaml exists${NC}"

    # Validate azure.yaml syntax
    if ! python3 -c "import yaml; yaml.safe_load(open('azure.yaml'))" 2>/dev/null; then
        echo -e "${RED}❌ azure.yaml syntax error${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ azure.yaml syntax valid${NC}"

    # Check required sections
    if ! grep -q "^name:" azure.yaml; then
        echo -e "${RED}❌ azure.yaml missing 'name' field${NC}"
        return 1
    fi

    if ! grep -q "^services:" azure.yaml; then
        echo -e "${RED}❌ azure.yaml missing 'services' section${NC}"
        return 1
    fi

    if ! grep -q "^infra:" azure.yaml; then
        echo -e "${RED}❌ azure.yaml missing 'infra' section${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ azure.yaml structure valid${NC}"
    return 0
}

test_environment_setup() {
    echo -e "${BLUE}🔍 Testing environment setup...${NC}"

    # Test environment setup script
    if [ ! -f "scripts/setup-environments.sh" ]; then
        echo -e "${RED}❌ Environment setup script not found${NC}"
        return 1
    fi

    if [ ! -x "scripts/setup-environments.sh" ]; then
        echo -e "${RED}❌ Environment setup script not executable${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Environment setup script ready${NC}"

    # Test configuration update script
    if [ ! -f "scripts/update-env-from-deployment.sh" ]; then
        echo -e "${RED}❌ Configuration update script not found${NC}"
        return 1
    fi

    if [ ! -x "scripts/update-env-from-deployment.sh" ]; then
        echo -e "${RED}❌ Configuration update script not executable${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Configuration update script ready${NC}"
    return 0
}

test_backend_dockerfile() {
    echo -e "${BLUE}🔍 Testing backend Docker configuration...${NC}"

    if [ ! -f "backend/Dockerfile" ]; then
        echo -e "${RED}❌ Backend Dockerfile not found${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ Backend Dockerfile exists${NC}"

    # Check if requirements.txt exists
    if [ ! -f "backend/requirements.txt" ]; then
        echo -e "${RED}❌ Backend requirements.txt not found${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ Backend requirements.txt exists${NC}"

    return 0
}

dry_run_deployment() {
    echo -e "${BLUE}🔍 Testing deployment dry run...${NC}"

    # Create test environment if it doesn't exist
    if ! azd env list | grep -q "test "; then
        echo "Creating test environment..."
        azd env new test --no-prompt
        azd env set AZURE_LOCATION eastus
    fi

    azd env select test

    # Test what-if deployment
    echo "Running deployment what-if analysis..."
    if azd provision --preview &> /dev/null; then
        echo -e "${GREEN}✅ Deployment preview successful${NC}"
    else
        echo -e "${RED}❌ Deployment preview failed${NC}"
        return 1
    fi

    return 0
}

# Run all tests
main() {
    echo -e "${BLUE}🚀 Starting Azure Universal RAG Infrastructure Tests${NC}"
    echo ""

    TESTS_PASSED=0
    TESTS_FAILED=0

    # Test 1: azd setup
    if test_azd_setup; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    echo ""

    # Test 2: Bicep syntax
    if test_bicep_syntax; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    echo ""

    # Test 3: azd configuration
    if test_azd_config; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    echo ""

    # Test 4: Environment setup
    if test_environment_setup; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    echo ""

    # Test 5: Backend Docker configuration
    if test_backend_dockerfile; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    echo ""

    # Test 6: Dry run deployment (optional, requires Azure access)
    if [ "${1:-}" = "--full" ]; then
        if dry_run_deployment; then
            ((TESTS_PASSED++))
        else
            ((TESTS_FAILED++))
        fi
        echo ""
    fi

    # Summary
    echo -e "${BLUE}📊 Test Results:${NC}"
    echo -e "${GREEN}✅ Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}❌ Failed: $TESTS_FAILED${NC}"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! Infrastructure is ready for deployment.${NC}"
        echo ""
        echo -e "${BLUE}📋 Next steps:${NC}"
        echo "  1. Setup environments: ./scripts/setup-environments.sh"
        echo "  2. Select environment: azd env select development"
        echo "  3. Deploy infrastructure: azd up"
        return 0
    else
        echo -e "${RED}❌ Some tests failed. Please fix the issues before deployment.${NC}"
        return 1
    fi
}

# Show help
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Azure Universal RAG Infrastructure Testing Script"
    echo ""
    echo "Usage:"
    echo "  $0              Run basic infrastructure tests"
    echo "  $0 --full       Run all tests including deployment dry run"
    echo "  $0 --help       Show this help message"
    echo ""
    echo "Tests:"
    echo "  1. Azure Developer CLI setup"
    echo "  2. Bicep template syntax validation"
    echo "  3. azd configuration validation"
    echo "  4. Environment setup scripts"
    echo "  5. Backend Docker configuration"
    echo "  6. Deployment dry run (--full only)"
    exit 0
fi

# Run tests
main "$@"
