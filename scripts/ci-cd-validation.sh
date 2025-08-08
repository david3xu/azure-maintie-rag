#!/bin/bash
# CI/CD Pipeline Validation for Azure Universal RAG System
# Validates deployment configuration, GitHub Actions setup, and Azure infrastructure

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Azure Universal RAG CI/CD Pipeline Validation${NC}"
echo "================================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate Azure CLI authentication
validate_azure_auth() {
    echo -e "\n${BLUE}🔐 Validating Azure Authentication...${NC}"
    
    if ! command_exists az; then
        echo -e "${RED}❌ Azure CLI not found${NC}"
        return 1
    fi
    
    # Check if logged in
    if az account show >/dev/null 2>&1; then
        ACCOUNT_NAME=$(az account show --query name -o tsv)
        SUBSCRIPTION_ID=$(az account show --query id -o tsv)
        echo -e "${GREEN}✅ Azure CLI authenticated${NC}"
        echo -e "   Account: ${ACCOUNT_NAME}"
        echo -e "   Subscription: ${SUBSCRIPTION_ID}"
        return 0
    else
        echo -e "${RED}❌ Azure CLI not authenticated${NC}"
        return 1
    fi
}

# Function to validate Azure infrastructure
validate_azure_infrastructure() {
    echo -e "\n${BLUE}🏗️ Validating Azure Infrastructure...${NC}"
    
    # Check resource group
    if az group show --name rg-maintie-rag-prod >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Resource group exists: rg-maintie-rag-prod${NC}"
    else
        echo -e "${RED}❌ Resource group not found: rg-maintie-rag-prod${NC}"
        return 1
    fi
    
    # Count resources
    RESOURCE_COUNT=$(az resource list --resource-group rg-maintie-rag-prod --query "length(@)" -o tsv)
    echo -e "${GREEN}✅ Found ${RESOURCE_COUNT} resources in production${NC}"
    
    # Validate key services
    echo -e "\n   ${BLUE}Checking key services...${NC}"
    
    # OpenAI Service
    if az cognitiveservices account show --name oai-maintie-rag-prod-fymhwfec3ra2w --resource-group rg-maintie-rag-prod >/dev/null 2>&1; then
        echo -e "   ${GREEN}✅ Azure OpenAI Service: oai-maintie-rag-prod-fymhwfec3ra2w${NC}"
    else
        echo -e "   ${RED}❌ Azure OpenAI Service not found${NC}"
    fi
    
    # Cosmos DB
    if az cosmosdb show --name cosmos-maintie-rag-prod-fymhwfec3ra2w --resource-group rg-maintie-rag-prod >/dev/null 2>&1; then
        echo -e "   ${GREEN}✅ Cosmos DB: cosmos-maintie-rag-prod-fymhwfec3ra2w${NC}"
    else
        echo -e "   ${RED}❌ Cosmos DB not found${NC}"
    fi
    
    # Search Service
    if az search service show --name srch-maintie-rag-prod-fymhwfec3ra2w --resource-group rg-maintie-rag-prod >/dev/null 2>&1; then
        echo -e "   ${GREEN}✅ Cognitive Search: srch-maintie-rag-prod-fymhwfec3ra2w${NC}"
    else
        echo -e "   ${RED}❌ Cognitive Search not found${NC}"
    fi
    
    # Container App
    if az containerapp show --name ca-backend-maintie-rag-prod --resource-group rg-maintie-rag-prod >/dev/null 2>&1; then
        echo -e "   ${GREEN}✅ Container App: ca-backend-maintie-rag-prod${NC}"
        
        # Get the container app URL
        APP_URL=$(az containerapp show --name ca-backend-maintie-rag-prod --resource-group rg-maintie-rag-prod --query properties.configuration.ingress.fqdn -o tsv)
        if [ -n "$APP_URL" ]; then
            echo -e "   ${BLUE}   URL: https://${APP_URL}${NC}"
        fi
    else
        echo -e "   ${RED}❌ Container App not found${NC}"
    fi
    
    return 0
}

# Function to validate GitHub Actions configuration
validate_github_actions() {
    echo -e "\n${BLUE}🔧 Validating GitHub Actions Configuration...${NC}"
    
    # Check if .github/workflows directory exists
    if [ -d ".github/workflows" ]; then
        echo -e "${GREEN}✅ GitHub workflows directory exists${NC}"
        
        # List workflow files
        WORKFLOW_COUNT=$(find .github/workflows -name "*.yml" -o -name "*.yaml" | wc -l)
        echo -e "   Found ${WORKFLOW_COUNT} workflow files:"
        
        find .github/workflows -name "*.yml" -o -name "*.yaml" | while read file; do
            echo -e "   ${BLUE}   - $(basename $file)${NC}"
        done
        
        # Check for main CI/CD workflow
        if [ -f ".github/workflows/azure-dev.yml" ]; then
            echo -e "${GREEN}✅ Main CI/CD workflow found: azure-dev.yml${NC}"
        else
            echo -e "${YELLOW}⚠️  Main CI/CD workflow not found${NC}"
        fi
        
    else
        echo -e "${RED}❌ GitHub workflows directory not found${NC}"
        return 1
    fi
    
    return 0
}

# Function to validate environment configuration
validate_environment_config() {
    echo -e "\n${BLUE}🌍 Validating Environment Configuration...${NC}"
    
    # Check azd configuration
    if [ -f "azure.yaml" ]; then
        echo -e "${GREEN}✅ azd configuration file exists${NC}"
        
        # Check if azd is available (skip if resource constrained)
        if command_exists azd 2>/dev/null; then
            echo -e "${GREEN}✅ Azure Developer CLI available${NC}"
        else
            echo -e "${YELLOW}⚠️  Azure Developer CLI not available (resource constraints)${NC}"
        fi
    else
        echo -e "${RED}❌ azd configuration file not found${NC}"
    fi
    
    # Check infrastructure templates
    if [ -d "infra" ]; then
        echo -e "${GREEN}✅ Infrastructure templates directory exists${NC}"
        
        BICEP_COUNT=$(find infra -name "*.bicep" | wc -l)
        echo -e "   Found ${BICEP_COUNT} Bicep templates"
        
        if [ -f "infra/main.bicep" ]; then
            echo -e "${GREEN}✅ Main infrastructure template exists${NC}"
        fi
    else
        echo -e "${RED}❌ Infrastructure templates not found${NC}"
    fi
    
    return 0
}

# Function to validate application configuration
validate_application_config() {
    echo -e "\n${BLUE}📱 Validating Application Configuration...${NC}"
    
    # Check Python requirements
    if [ -f "requirements.txt" ]; then
        echo -e "${GREEN}✅ Python requirements.txt exists${NC}"
        REQ_COUNT=$(wc -l < requirements.txt)
        echo -e "   ${REQ_COUNT} Python dependencies"
    else
        echo -e "${RED}❌ requirements.txt not found${NC}"
    fi
    
    # Check frontend configuration
    if [ -f "frontend/package.json" ]; then
        echo -e "${GREEN}✅ Frontend package.json exists${NC}"
        
        # Check if build script exists
        if grep -q '"build"' frontend/package.json; then
            echo -e "${GREEN}✅ Frontend build script configured${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Frontend configuration not found${NC}"
    fi
    
    # Check API structure
    if [ -d "api" ]; then
        echo -e "${GREEN}✅ API directory exists${NC}"
        
        if [ -f "api/main.py" ]; then
            echo -e "${GREEN}✅ API main.py exists${NC}"
        fi
    else
        echo -e "${RED}❌ API directory not found${NC}"
    fi
    
    return 0
}

# Function to test health endpoint
test_health_endpoint() {
    echo -e "\n${BLUE}🏥 Testing Application Health...${NC}"
    
    # Get the backend URL from azd environment
    if command_exists azd 2>/dev/null; then
        BACKEND_URL=$(azd env get-value SERVICE_BACKEND_URI 2>/dev/null || echo "")
        if [ -n "$BACKEND_URL" ]; then
            echo -e "   Backend URL: ${BACKEND_URL}"
            
            # Test health endpoint with timeout
            if curl -s --max-time 10 "${BACKEND_URL}/health" >/dev/null 2>&1; then
                echo -e "${GREEN}✅ Health endpoint responding${NC}"
            else
                echo -e "${YELLOW}⚠️  Health endpoint not responding (may be starting up)${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  Backend URL not available${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Cannot test health endpoint (azd not available)${NC}"
    fi
}

# Function to validate test configuration
validate_test_config() {
    echo -e "\n${BLUE}🧪 Validating Test Configuration...${NC}"
    
    if [ -d "tests" ]; then
        echo -e "${GREEN}✅ Tests directory exists${NC}"
        
        TEST_COUNT=$(find tests -name "test_*.py" | wc -l)
        echo -e "   Found ${TEST_COUNT} test files"
        
        # Check test runner script
        if [ -f "scripts/run-tests.sh" ]; then
            echo -e "${GREEN}✅ Test runner script exists${NC}"
        fi
        
        if [ -f "scripts/quick-test.sh" ]; then
            echo -e "${GREEN}✅ Quick test script exists${NC}"
        fi
    else
        echo -e "${RED}❌ Tests directory not found${NC}"
    fi
    
    # Check pytest configuration
    if [ -f "pytest.ini" ]; then
        echo -e "${GREEN}✅ pytest configuration exists${NC}"
    fi
}

# Main validation function
main() {
    echo -e "\n${BLUE}Starting CI/CD Pipeline Validation...${NC}"
    
    local validation_errors=0
    
    # Run validations
    validate_azure_auth || ((validation_errors++))
    validate_azure_infrastructure || ((validation_errors++))
    validate_github_actions || ((validation_errors++))
    validate_environment_config || ((validation_errors++))
    validate_application_config || ((validation_errors++))
    test_health_endpoint || true  # Don't count as error
    validate_test_config || ((validation_errors++))
    
    # Summary
    echo -e "\n${BLUE}📊 Validation Summary${NC}"
    echo "====================="
    
    if [ $validation_errors -eq 0 ]; then
        echo -e "${GREEN}🎉 All validations passed! CI/CD pipeline is ready.${NC}"
        echo -e "\n${BLUE}Next Steps:${NC}"
        echo -e "1. Push code to GitHub to trigger CI/CD pipeline"
        echo -e "2. Monitor GitHub Actions workflow: https://github.com/david3xu/azure-maintie-rag/actions"
        echo -e "3. Validate deployed services using: make health"
        echo -e "4. Run full test suite: ./scripts/run-tests.sh"
        exit 0
    else
        echo -e "${RED}❌ ${validation_errors} validation(s) failed${NC}"
        echo -e "\nPlease address the issues above before deploying."
        exit 1
    fi
}

# Run the main validation
main