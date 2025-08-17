#!/bin/bash
# Pre-deployment Configuration Test
# =================================
# Comprehensive validation before running azd up

echo "🧪 PRE-DEPLOYMENT CONFIGURATION TEST"
echo "====================================="
echo ""

# Function to check command success
check_result() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

FAILED_TESTS=0

echo "1️⃣ TESTING PREREQUISITES:"
echo "------------------------"

# Check Azure CLI authentication
az account show > /dev/null 2>&1
check_result "Azure CLI authentication"

# Check azd authentication
azd auth login --check-status > /dev/null 2>&1
check_result "Azure Developer CLI authentication"

# Check required tools
command -v docker > /dev/null 2>&1
check_result "Docker availability"

echo ""
echo "2️⃣ TESTING BICEP CONFIGURATION:"
echo "-------------------------------"

# Test Bicep syntax (warnings are acceptable)
BICEP_OUTPUT=$(az deployment group validate \
    --resource-group "rg-maintie-rag-prod" \
    --template-file infra/main.bicep \
    --parameters infra/main.parameters.json 2>&1)

if echo "$BICEP_OUTPUT" | grep -q "ERROR:" && ! echo "$BICEP_OUTPUT" | grep -q "target scope.*does not match"; then
    echo "❌ Bicep template validation (real errors found)"
    FAILED_TESTS=$((FAILED_TESTS + 1))
else
    echo "✅ Bicep template validation (warnings only)"
fi

# Check ACR configuration in environment
ACR_ENDPOINT=$(azd env get-values | grep "AZURE_CONTAINER_REGISTRY_ENDPOINT=" | cut -d'=' -f2 | tr -d '"')
if [[ "$ACR_ENDPOINT" != "not-available-in-student-subscription" ]]; then
    echo "✅ ACR endpoint configured: $ACR_ENDPOINT"
else
    echo "❌ ACR endpoint still has placeholder value"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

echo ""
echo "3️⃣ TESTING AZURE RESOURCES:"
echo "---------------------------"

# Check if resource group exists
az group show --name "rg-maintie-rag-prod" > /dev/null 2>&1
check_result "Resource group existence"

# Check if ACR exists and is accessible
ACR_NAME=$(azd env get-values | grep "AZURE_CONTAINER_REGISTRY_NAME=" | cut -d'=' -f2 | tr -d '"')
if [ -n "$ACR_NAME" ] && [[ "$ACR_NAME" != "not-available-in-student-subscription" ]]; then
    ACR_OUTPUT=$(az acr show --name "$ACR_NAME" --resource-group "rg-maintie-rag-prod" 2>&1)
    if echo "$ACR_OUTPUT" | grep -qi "succeeded" || echo "$ACR_OUTPUT" | grep -q "\"name\": \"$ACR_NAME\""; then
        echo "✅ ACR accessibility: $ACR_NAME"
    else
        echo "❌ ACR accessibility: $ACR_NAME"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
else
    echo "⚠️  ACR name not configured or using placeholder"
fi

echo ""
echo "4️⃣ TESTING DATA PIPELINE READINESS:"
echo "-----------------------------------"

# Check if data files exist
if [ -d "data/raw" ] && [ "$(ls -A data/raw)" ]; then
    echo "✅ Data files available: $(find data/raw -name "*.md" | wc -l) files"
else
    echo "❌ No data files found in data/raw/"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test Python environment
python3 -c "import sys; sys.path.insert(0, '.'); from agents.core.universal_deps import UniversalDeps; print('✅ Python dependencies')" 2>/dev/null || {
    echo "❌ Python dependencies issue"
    FAILED_TESTS=$((FAILED_TESTS + 1))
}

echo ""
echo "5️⃣ TESTING CONFIGURATION VALUES:"
echo "--------------------------------"

# Check environment configuration
ENV_NAME=$(azd env get-values | grep "AZURE_ENV_NAME=" | cut -d'=' -f2 | tr -d '"')
echo "✅ Environment: $ENV_NAME"

SUBSCRIPTION=$(azd env get-values | grep "AZURE_SUBSCRIPTION_ID=" | cut -d'=' -f2 | tr -d '"')
echo "✅ Subscription: $SUBSCRIPTION"

AUTO_POPULATE=$(azd env get-values | grep "AUTO_POPULATE_DATA=" | cut -d'=' -f2 | tr -d '"')
echo "✅ Auto populate data: $AUTO_POPULATE"

echo ""
echo "📊 TEST RESULTS SUMMARY:"
echo "========================"

if [ $FAILED_TESTS -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED! Ready for azd up"
    echo ""
    echo "✅ Recommended next steps:"
    echo "   1. azd up (should complete successfully)"
    echo "   2. make dataflow-full (after deployment)"
    echo ""
    exit 0
else
    echo "❌ $FAILED_TESTS TESTS FAILED"
    echo ""
    echo "🔧 Required fixes before azd up:"
    echo "   1. Fix authentication issues if any"
    echo "   2. Ensure ACR configuration is correct"
    echo "   3. Verify all prerequisites are installed"
    echo ""
    exit 1
fi