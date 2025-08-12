#!/bin/bash

# Azure Universal RAG Deployment Monitor
# Real-time monitoring of deployment status and health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=5174
REFRESH_INTERVAL=10

echo -e "${BLUE}🔍 Azure Universal RAG Deployment Monitor${NC}"
echo "=========================================="
echo ""

# Function to check service status
check_service() {
    local service_name=$1
    local check_command=$2
    
    if eval $check_command &> /dev/null; then
        echo -e "${GREEN}✅ $service_name: Running${NC}"
        return 0
    else
        echo -e "${RED}❌ $service_name: Not running${NC}"
        return 1
    fi
}

# Function to check Azure resources
check_azure_resources() {
    echo -e "\n${BLUE}☁️  Azure Resources Status:${NC}"
    
    # Check if logged in to Azure
    if az account show &> /dev/null; then
        SUBSCRIPTION=$(az account show --query name -o tsv)
        echo -e "  📋 Subscription: $SUBSCRIPTION"
        
        # Check resource group
        RG_NAME="rg-maintie-rag-prod"
        if az group show --name $RG_NAME &> /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ Resource Group: $RG_NAME exists${NC}"
            
            # Count resources
            RESOURCE_COUNT=$(az resource list --resource-group $RG_NAME --query "length(@)" -o tsv 2>/dev/null || echo "0")
            echo -e "  📊 Resources deployed: $RESOURCE_COUNT"
            
            # Check key services
            echo -e "\n  ${BLUE}Key Services:${NC}"
            
            # Azure OpenAI
            if az cognitiveservices account list --resource-group $RG_NAME --query "[?kind=='OpenAI'].name" -o tsv | grep -q .; then
                echo -e "    ${GREEN}✅ Azure OpenAI${NC}"
            else
                echo -e "    ${YELLOW}⚠️  Azure OpenAI not found${NC}"
            fi
            
            # Cognitive Search
            if az search service list --resource-group $RG_NAME --query "[].name" -o tsv | grep -q .; then
                echo -e "    ${GREEN}✅ Azure Cognitive Search${NC}"
            else
                echo -e "    ${YELLOW}⚠️  Azure Cognitive Search not found${NC}"
            fi
            
            # Cosmos DB
            if az cosmosdb list --resource-group $RG_NAME --query "[].name" -o tsv | grep -q .; then
                echo -e "    ${GREEN}✅ Azure Cosmos DB${NC}"
            else
                echo -e "    ${YELLOW}⚠️  Azure Cosmos DB not found${NC}"
            fi
            
            # Storage Account
            if az storage account list --resource-group $RG_NAME --query "[].name" -o tsv | grep -q .; then
                echo -e "    ${GREEN}✅ Azure Storage${NC}"
            else
                echo -e "    ${YELLOW}⚠️  Azure Storage not found${NC}"
            fi
            
        else
            echo -e "  ${YELLOW}⚠️  Resource Group: $RG_NAME not found${NC}"
            echo -e "  💡 Run 'azd up' to deploy infrastructure"
        fi
    else
        echo -e "  ${RED}❌ Not logged in to Azure${NC}"
        echo -e "  💡 Run 'az login' to authenticate"
    fi
}

# Function to check local services
check_local_services() {
    echo -e "\n${BLUE}🖥️  Local Services Status:${NC}"
    
    # Backend API
    if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✅ Backend API (port $BACKEND_PORT)${NC}"
        
        # Get detailed health
        HEALTH=$(curl -s http://localhost:$BACKEND_PORT/health)
        if [ ! -z "$HEALTH" ]; then
            STATUS=$(echo $HEALTH | jq -r '.status' 2>/dev/null || echo "unknown")
            SERVICES=$(echo $HEALTH | jq -r '.services_available[]' 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
            echo -e "    Status: $STATUS"
            echo -e "    Services: $SERVICES"
        fi
    else
        echo -e "  ${RED}❌ Backend API (port $BACKEND_PORT)${NC}"
        echo -e "    💡 Run 'uvicorn api.main:app --port $BACKEND_PORT --reload' to start"
    fi
    
    # Frontend
    if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
        echo -e "  ${GREEN}✅ Frontend (port $FRONTEND_PORT)${NC}"
    else
        echo -e "  ${RED}❌ Frontend (port $FRONTEND_PORT)${NC}"
        echo -e "    💡 Run 'cd frontend && npm run dev' to start"
    fi
    
    # Check Python processes
    PYTHON_PROCS=$(ps aux | grep -E "python.*azure-maintie-rag" | grep -v grep | wc -l)
    if [ $PYTHON_PROCS -gt 0 ]; then
        echo -e "  📊 Python processes: $PYTHON_PROCS running"
    fi
    
    # Check Node processes
    NODE_PROCS=$(ps aux | grep -E "node.*vite" | grep -v grep | wc -l)
    if [ $NODE_PROCS -gt 0 ]; then
        echo -e "  📊 Node processes: $NODE_PROCS running"
    fi
}

# Function to check GitHub Actions
check_github_actions() {
    echo -e "\n${BLUE}🔄 GitHub Actions Status:${NC}"
    
    if command -v gh &> /dev/null; then
        if gh auth status &> /dev/null 2>&1; then
            # Get latest workflow runs
            LATEST_RUN=$(gh run list --workflow=azure-dev.yml --limit 1 --json status,conclusion,databaseId,name 2>/dev/null)
            
            if [ ! -z "$LATEST_RUN" ] && [ "$LATEST_RUN" != "[]" ]; then
                STATUS=$(echo $LATEST_RUN | jq -r '.[0].status' 2>/dev/null)
                CONCLUSION=$(echo $LATEST_RUN | jq -r '.[0].conclusion' 2>/dev/null)
                RUN_ID=$(echo $LATEST_RUN | jq -r '.[0].databaseId' 2>/dev/null)
                
                if [ "$STATUS" = "in_progress" ]; then
                    echo -e "  ${YELLOW}🔄 Deployment in progress (Run #$RUN_ID)${NC}"
                elif [ "$CONCLUSION" = "success" ]; then
                    echo -e "  ${GREEN}✅ Last deployment successful (Run #$RUN_ID)${NC}"
                elif [ "$CONCLUSION" = "failure" ]; then
                    echo -e "  ${RED}❌ Last deployment failed (Run #$RUN_ID)${NC}"
                    echo -e "    💡 Run 'gh run view $RUN_ID' for details"
                else
                    echo -e "  Status: $STATUS / $CONCLUSION"
                fi
            else
                echo -e "  ${YELLOW}⚠️  No workflow runs found${NC}"
            fi
        else
            echo -e "  ${YELLOW}⚠️  GitHub CLI not authenticated${NC}"
            echo -e "    💡 Run 'gh auth login' to authenticate"
        fi
    else
        echo -e "  ${YELLOW}⚠️  GitHub CLI not installed${NC}"
        echo -e "    💡 Install from: https://cli.github.com/"
    fi
}

# Function to check system resources
check_system_resources() {
    echo -e "\n${BLUE}💻 System Resources:${NC}"
    
    # Memory usage
    MEM_USAGE=$(free -h | grep Mem | awk '{print $3 "/" $2}')
    echo -e "  Memory: $MEM_USAGE"
    
    # Disk usage
    DISK_USAGE=$(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')
    echo -e "  Disk: $DISK_USAGE"
    
    # Load average
    LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}')
    echo -e "  Load average:$LOAD_AVG"
}

# Function to show deployment progress
show_deployment_progress() {
    echo -e "\n${BLUE}📊 Deployment Checklist:${NC}"
    
    local checks_passed=0
    local total_checks=8
    
    # Check prerequisites
    echo -e "\n  Prerequisites:"
    if command -v az &> /dev/null; then
        echo -e "    ${GREEN}✅ Azure CLI installed${NC}"
        ((checks_passed++))
    else
        echo -e "    ${RED}❌ Azure CLI not installed${NC}"
    fi
    
    if command -v azd &> /dev/null; then
        echo -e "    ${GREEN}✅ Azure Developer CLI installed${NC}"
        ((checks_passed++))
    else
        echo -e "    ${RED}❌ Azure Developer CLI not installed${NC}"
    fi
    
    if az account show &> /dev/null; then
        echo -e "    ${GREEN}✅ Azure authenticated${NC}"
        ((checks_passed++))
    else
        echo -e "    ${RED}❌ Azure not authenticated${NC}"
    fi
    
    # Check deployment status
    echo -e "\n  Deployment:"
    if [ -f ".azure/prod/.env" ]; then
        echo -e "    ${GREEN}✅ Environment configured${NC}"
        ((checks_passed++))
    else
        echo -e "    ${YELLOW}⚠️  Environment not configured${NC}"
    fi
    
    RG_NAME="rg-maintie-rag-prod"
    if az group show --name $RG_NAME &> /dev/null 2>&1; then
        echo -e "    ${GREEN}✅ Infrastructure deployed${NC}"
        ((checks_passed++))
    else
        echo -e "    ${YELLOW}⚠️  Infrastructure not deployed${NC}"
    fi
    
    # Check services
    echo -e "\n  Services:"
    if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
        echo -e "    ${GREEN}✅ Backend running${NC}"
        ((checks_passed++))
    else
        echo -e "    ${YELLOW}⚠️  Backend not running${NC}"
    fi
    
    if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
        echo -e "    ${GREEN}✅ Frontend running${NC}"
        ((checks_passed++))
    else
        echo -e "    ${YELLOW}⚠️  Frontend not running${NC}"
    fi
    
    # CI/CD
    if [ -f ".github/workflows/azure-dev.yml" ]; then
        echo -e "    ${GREEN}✅ CI/CD configured${NC}"
        ((checks_passed++))
    else
        echo -e "    ${RED}❌ CI/CD not configured${NC}"
    fi
    
    # Progress bar
    echo -e "\n  ${BLUE}Overall Progress:${NC}"
    PERCENTAGE=$((checks_passed * 100 / total_checks))
    echo -n "  ["
    for i in $(seq 1 10); do
        if [ $((i * 10)) -le $PERCENTAGE ]; then
            echo -n "█"
        else
            echo -n "░"
        fi
    done
    echo "] $PERCENTAGE% ($checks_passed/$total_checks)"
    
    # Next steps
    if [ $checks_passed -lt $total_checks ]; then
        echo -e "\n${YELLOW}📝 Next Steps:${NC}"
        
        if ! command -v az &> /dev/null; then
            echo "  1. Install Azure CLI: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
        fi
        
        if ! command -v azd &> /dev/null; then
            echo "  2. Install Azure Developer CLI: curl -fsSL https://aka.ms/install-azd.sh | bash"
        fi
        
        if ! az account show &> /dev/null; then
            echo "  3. Login to Azure: az login"
        fi
        
        if ! az group show --name $RG_NAME &> /dev/null 2>&1; then
            echo "  4. Deploy infrastructure: azd up"
        fi
        
        if ! curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
            echo "  5. Start backend: uvicorn api.main:app --port $BACKEND_PORT --reload"
        fi
        
        if ! curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
            echo "  6. Start frontend: cd frontend && npm run dev"
        fi
    else
        echo -e "\n${GREEN}🎉 All checks passed! System fully deployed.${NC}"
    fi
}

# Main monitoring loop
main() {
    while true; do
        clear
        echo -e "${BLUE}🔍 Azure Universal RAG Deployment Monitor${NC}"
        echo "=========================================="
        echo "$(date '+%Y-%m-%d %H:%M:%S')"
        
        check_local_services
        check_azure_resources
        check_github_actions
        check_system_resources
        show_deployment_progress
        
        echo -e "\n${BLUE}Refreshing in $REFRESH_INTERVAL seconds... (Ctrl+C to exit)${NC}"
        sleep $REFRESH_INTERVAL
    done
}

# Handle single run vs continuous monitoring
if [ "$1" = "--once" ]; then
    check_local_services
    check_azure_resources
    check_github_actions
    check_system_resources
    show_deployment_progress
else
    main
fi