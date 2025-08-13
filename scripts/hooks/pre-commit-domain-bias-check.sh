#!/bin/bash
#
# Universal RAG Domain Bias Detection Hook (Simplified)
# ====================================================
#
# Enforces zero-domain-bias philosophy by detecting hardcoded domain assumptions.
# Philosophy: DISCOVER domain characteristics, don't assume them.

set -euo pipefail

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VIOLATION_COUNT=0
CHECK_PATHS=("agents" "config" "scripts/dataflow" "infrastructure")

echo -e "${BLUE}🔍 Universal RAG Domain Bias Check${NC}"
echo "=================================="

# Function to report violations
report_violation() {
    local file="$1"
    local line_num="$2"
    local content="$3"
    
    echo -e "${RED}❌ DOMAIN BIAS${NC} in ${file}:${line_num}"
    echo -e "   ${content}"
    echo ""
    
    VIOLATION_COUNT=$((VIOLATION_COUNT + 1))
}

# Function to check for critical domain bias patterns
check_domain_bias() {
    local file="$1"
    
    # Core domain bias patterns (simplified but effective)
    local patterns=(
        # Hardcoded domain types
        '"technical"' '"academic"' '"legal"' '"medical"' '"financial"' '"business"'
        "'technical'" "'academic'" "'legal'" "'medical'" "'financial'" "'business'"
        
        # Domain classification logic
        'if.*domain.*==' 'elif.*domain.*==' 'classify_domain' 'detect_domain'
        
        # Predetermined processing
        'technical_processing' 'medical_processing' 'legal_processing'
        'process_technical' 'process_medical' 'process_legal'
        
        # Mock/fake patterns (production quality)
        'mock.*data' 'fake.*data' 'simulated.*data'
        'return.*mock' 'return.*fake' 'return.*simulated'
    )
    
    for pattern in "${patterns[@]}"; do
        while IFS=: read -r line_num content; do
            if [[ -n "$content" ]]; then
                # Skip comments and documentation
                if [[ "$content" =~ ^[[:space:]]*# ]] || [[ "$content" =~ ^[[:space:]]*\"\"\" ]]; then
                    continue
                fi
                
                # Skip legitimate infrastructure patterns
                if [[ "$content" =~ "if not azure_" ]] || [[ "$content" =~ "validation\[" ]]; then
                    continue
                fi
                
                report_violation "$file" "$line_num" "$(echo "$content" | sed 's/^[[:space:]]*//')"
            fi
        done < <(grep -n -E "$pattern" "$file" 2>/dev/null || true)
    done
}

# Main execution - simplified and fast
echo "Scanning for domain bias violations..."
echo ""

# Check all relevant Python files
for check_path in "${CHECK_PATHS[@]}"; do
    if [[ -d "$check_path" ]]; then
        echo -e "${BLUE}📂 Checking ${check_path}/${NC}"
        
        while IFS= read -r -d '' file; do
            # Skip __pycache__ and .git directories
            if [[ "$file" =~ __pycache__ ]] || [[ "$file" =~ \.git ]]; then
                continue
            fi
            
            echo -e "   📄 $(basename "$file")"
            check_domain_bias "$file"
            
        done < <(find "$check_path" -name "*.py" -type f -print0 2>/dev/null || true)
    fi
done

echo ""
echo "========================================"

# Report results
if [ $VIOLATION_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: No domain bias violations detected${NC}"
    echo -e "${GREEN}🌍 Universal RAG philosophy maintained${NC}"
    exit 0
else
    echo -e "${RED}❌ FAIL: Found $VIOLATION_COUNT domain bias violations${NC}"
    echo ""
    echo -e "Universal RAG Principles:"
    echo "   • DISCOVER domain characteristics from content"
    echo "   • ADAPT parameters based on measured properties"
    echo "   • AVOID predetermined domain categories"
    echo "   • USE content-agnostic patterns and algorithms"
    echo ""
    exit 1
fi
