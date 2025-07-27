#!/bin/bash

# Azure Universal RAG - 10% Lifecycle Test Runner
# This script runs a complete end-to-end test using 10% sample data

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Azure Universal RAG - 10% Lifecycle Test${NC}"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: Must run from backend/ directory${NC}"
    echo "Usage: cd backend && bash scripts/organized/run_lifecycle_test.sh"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  No virtual environment found. Creating one...${NC}"
    python -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/.requirements_installed" ]; then
    echo -e "${BLUE}🔄 Installing requirements...${NC}"
    pip install -r requirements.txt
    touch venv/.requirements_installed
fi

# Check Azure configuration
echo -e "${BLUE}🔄 Validating Azure configuration...${NC}"
if python scripts/organized/azure_services/azure_config_validator.py; then
    echo -e "${GREEN}✅ Azure configuration valid${NC}"
else
    echo -e "${RED}❌ Azure configuration invalid. Please check your .env file${NC}"
    exit 1
fi

# Check data state
echo -e "${BLUE}🔄 Checking Azure data state...${NC}"
python scripts/organized/azure_services/azure_data_state.py

# Check if 10% sample data exists
SAMPLE_FILE="data/raw/demo_sample_10percent.md"
if [ ! -f "$SAMPLE_FILE" ]; then
    echo -e "${RED}❌ Error: 10% sample data not found at $SAMPLE_FILE${NC}"
    echo "Please ensure the sample data file exists"
    exit 1
fi

echo -e "${GREEN}✅ 10% sample data found ($(wc -l < $SAMPLE_FILE) lines)${NC}"

# Run the lifecycle test
echo -e "${BLUE}🔄 Starting 10% lifecycle test...${NC}"
echo "This will take approximately 2-4 minutes"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the main lifecycle test (corrected version)
if python scripts/organized/workflows/lifecycle_test_10percent_corrected.py; then
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}🎉 Lifecycle test completed successfully!${NC}"
    echo -e "${GREEN}⏱️  Total duration: ${DURATION} seconds${NC}"
    
    # Show results location
    LATEST_RESULT=$(ls -t data/demo_outputs/lifecycle_test_10pct_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        echo -e "${BLUE}📊 Results saved to: $LATEST_RESULT${NC}"
        
        # Show summary
        echo ""
        echo -e "${BLUE}📈 Quick Summary:${NC}"
        python -c "
import json
try:
    with open('$LATEST_RESULT', 'r') as f:
        results = json.load(f)
    
    print(f\"  Session ID: {results.get('session_id', 'N/A')}\")
    print(f\"  Success Rate: {results.get('metrics', {}).get('success_rate', 0):.1%}\")
    print(f\"  Successful Stages: {results.get('metrics', {}).get('successful_stages', 0)}/{results.get('metrics', {}).get('total_stages', 0)}\")
    
    # Show stage durations
    stages = results.get('stages', {})
    print(f\"  \\nStage Durations:\")
    for stage, data in stages.items():
        status = data.get('status', 'unknown')
        duration = data.get('duration_seconds', 0)
        print(f\"    {stage}: {duration:.1f}s ({status})\")
        
except Exception as e:
    print(f\"  Could not parse results: {e}\")
"
    fi
    
    echo ""
    echo -e "${GREEN}🎯 Next Steps:${NC}"
    echo "  1. Review detailed results in the JSON file above"
    echo "  2. Check logs in logs/ directory for detailed execution info"
    echo "  3. Scale to full dataset if test successful"
    echo "  4. Run additional demos from scripts/organized/demos/"
    
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${RED}❌ Lifecycle test failed after ${DURATION} seconds${NC}"
    echo -e "${YELLOW}💡 Troubleshooting:${NC}"
    echo "  1. Check logs in logs/ directory"
    echo "  2. Verify Azure service connectivity"
    echo "  3. Check .env file configuration"
    echo "  4. Run azure_config_validator.py for detailed diagnostics"
    exit 1
fi

echo ""
echo -e "${BLUE}📚 For more information, see scripts/organized/README.md${NC}"