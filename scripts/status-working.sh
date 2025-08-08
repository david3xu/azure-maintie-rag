#!/bin/bash
# Azure Universal RAG - Status Working Script
# Simplified status check for Azure services

echo "🔍 Azure Universal RAG System Status"
echo "Session: $(date '+%Y%m%d_%H%M%S')"
echo "Timestamp: $(date)"
echo "=" * 60

# Check if we can run the Azure state check
if [ -f "scripts/dataflow/00_check_azure_state.py" ]; then
    echo "📊 Running Azure services validation..."
    python scripts/dataflow/00_check_azure_state.py --verbose
else
    echo "❌ Azure state check script not found"
fi

echo ""
echo "🌍 Current Environment Status:"
if [ -f ".azure/prod/.env" ]; then
    echo "   ✅ Production environment configuration found"
elif [ -f ".azure/staging/.env" ]; then
    echo "   ✅ Staging environment configuration found"
elif [ -f "config/environments/development.env" ]; then
    echo "   ✅ Development environment configuration found"
else
    echo "   ❌ No environment configuration found"
fi

echo ""
echo "🔐 Azure Authentication Status:"
if az account show >/dev/null 2>&1; then
    echo "   ✅ Azure CLI authenticated"
    az account show --query "name" -o tsv | sed 's/^/   📝 Active subscription: /'
else
    echo "   ❌ Azure CLI not authenticated - run 'az login'"
fi

echo ""
echo "📂 Data Directory Status:"
if [ -d "data/raw" ]; then
    file_count=$(find data/raw -type f | wc -l)
    echo "   ✅ Raw data directory found ($file_count files)"
else
    echo "   ❌ Raw data directory not found"
fi

echo ""
echo "Status check completed."