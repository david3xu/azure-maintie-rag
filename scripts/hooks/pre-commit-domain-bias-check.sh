#!/bin/bash
#
# Universal RAG Domain Bias Detection Hook
# =======================================
#
# This pre-commit hook enforces the Universal RAG zero-domain-bias philosophy by detecting:
# - Hardcoded domain categories (technical, academic, legal, etc.)
# - Predetermined domain assumptions and classifications
# - Non-universal patterns that violate domain-agnostic principles
# - Configuration biases and hardcoded domain logic
#
# Philosophy: The system should DISCOVER domain characteristics, not assume them.

set -euo pipefail

# ANSI color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
VIOLATION_COUNT=0
CHECK_PATHS=("agents" "config" "scripts/dataflow" "infrastructure")

echo -e "${BLUE}🔍 Universal RAG Domain Bias Detection${NC}"
echo "========================================"

# Function to report violations
report_violation() {
    local file="$1"
    local line_num="$2"
    local violation_type="$3"
    local content="$4"
    local explanation="$5"
    
    echo -e "${RED}❌ DOMAIN BIAS VIOLATION${NC} in ${PURPLE}${file}:${line_num}${NC}"
    echo -e "   ${YELLOW}Type:${NC} ${violation_type}"
    echo -e "   ${YELLOW}Code:${NC} ${content}"
    echo -e "   ${YELLOW}Issue:${NC} ${explanation}"
    echo ""
    
    VIOLATION_COUNT=$((VIOLATION_COUNT + 1))
}

# Function to check for hardcoded domain categories
check_hardcoded_domains() {
    local file="$1"
    
    # Domain category patterns that violate universal philosophy
    local domain_patterns=(
        # Predetermined domain types
        '"technical"'
        '"academic"'
        '"legal"'
        '"medical"'
        '"financial"'
        '"scientific"'
        '"business"'
        '"research"'
        '"educational"'
        "'technical'"
        "'academic'"
        "'legal'"
        "'medical'"
        "'financial'"
        "'scientific'"
        "'business'"
        "'research'"
        "'educational'"
        
        # Domain classification assumptions
        'domain_type.*=.*["'"'"']'
        'if.*domain.*==.*["'"'"']'
        'domain_category'
        'preset_domains'
        'known_domains'
        'DOMAIN_TYPES'
        'SUPPORTED_DOMAINS'
    )
    
    for pattern in "${domain_patterns[@]}"; do
        while IFS=: read -r line_num content; do
            if [[ -n "$content" ]]; then
                # Skip comments and documentation
                if [[ "$content" =~ ^[[:space:]]*# ]] || [[ "$content" =~ ^[[:space:]]*\"\"\" ]] || [[ "$content" =~ ^[[:space:]]*\'\'\' ]]; then
                    continue
                fi
                
                report_violation "$file" "$line_num" "Hardcoded Domain Category" \
                    "$(echo "$content" | sed 's/^[[:space:]]*//')" \
                    "Universal RAG should DISCOVER domains, not use predetermined categories"
            fi
        done < <(grep -n -E "$pattern" "$file" 2>/dev/null || true)
    done
}

# Function to check for domain-specific configuration bias
check_config_bias() {
    local file="$1"
    
    # Configuration patterns that violate domain-agnostic principles
    local config_bias_patterns=(
        # Technical vs non-technical assumptions
        'technical_vocabulary_ratio.*>.*[0-9]'
        'if.*technical.*density'
        'technical.*content'
        'academic.*document'
        'legal.*content'
        'medical.*terminology'
        
        # Hardcoded threshold adjustments based on domain assumptions
        'if.*domain.*technical'
        'if.*domain.*academic'
        'if.*technical_density.*>'
        'technical_indicators.*='
        'academic_indicators.*='
        
        # Predetermined entity types
        'expected_entity_types.*=.*\[.*["'"'"']CONCEPT["'"'"']'
        'technical_words.*=.*\['
        'academic_terms.*=.*\['
        'legal_terms.*=.*\['
    )
    
    for pattern in "${config_bias_patterns[@]}"; do
        while IFS=: read -r line_num content; do
            if [[ -n "$content" ]]; then
                # Skip comments
                if [[ "$content" =~ ^[[:space:]]*# ]]; then
                    continue
                fi
                
                report_violation "$file" "$line_num" "Configuration Domain Bias" \
                    "$(echo "$content" | sed 's/^[[:space:]]*//')" \
                    "Configuration should adapt based on DISCOVERED characteristics, not predetermined domain assumptions"
            fi
        done < <(grep -n -E "$pattern" "$file" 2>/dev/null || true)
    done
}

# Function to check for domain classification logic
check_domain_classification() {
    local file="$1"
    
    # Patterns that classify content into predetermined categories
    local classification_patterns=(
        # Domain classification functions
        'classify_domain'
        'detect_domain_type'
        'determine_domain'
        'identify_domain_category'
        
        # If-else chains for domain types
        'elif.*domain.*==.*["'"'"']'
        'if.*domain_type.*in'
        'domain_type.*in.*\['
        
        # Domain-specific processing branches
        'if.*is_technical'
        'if.*is_academic'
        'if.*is_legal'
        'process_technical'
        'process_academic'
        'handle_domain_type'
    )
    
    for pattern in "${classification_patterns[@]}"; do
        while IFS=: read -r line_num content; do
            if [[ -n "$content" ]]; then
                # Skip comments
                if [[ "$content" =~ ^[[:space:]]*# ]]; then
                    continue
                fi
                
                report_violation "$file" "$line_num" "Domain Classification Logic" \
                    "$(echo "$content" | sed 's/^[[:space:]]*//')" \
                    "Universal RAG should analyze content characteristics, not classify into predetermined domains"
            fi
        done < <(grep -n -E "$pattern" "$file" 2>/dev/null || true)
    done
}

# Function to check for non-universal patterns
check_non_universal_patterns() {
    local file="$1"
    
    # Patterns that violate universal design principles
    local non_universal_patterns=(
        # Hardcoded assumptions about content types
        'technical.*document'
        'academic.*paper'
        'legal.*document'
        'medical.*record'
        'scientific.*article'
        
        # Domain-specific extraction strategies
        'extract_technical'
        'extract_academic'
        'technical_extraction'
        'academic_extraction'
        
        # Hardcoded weights or parameters for specific domains
        'TECHNICAL_WEIGHT'
        'ACADEMIC_WEIGHT'
        'LEGAL_THRESHOLD'
        'MEDICAL_CONFIDENCE'
        
        # Domain-specific constants
        'TECHNICAL_VOCABULARY'
        'ACADEMIC_TERMS'
        'LEGAL_CONCEPTS'
    )
    
    for pattern in "${non_universal_patterns[@]}"; do
        while IFS=: read -r line_num content; do
            if [[ -n "$content" ]]; then
                # Skip comments and docstrings
                if [[ "$content" =~ ^[[:space:]]*# ]] || [[ "$content" =~ ^[[:space:]]*\"\"\" ]] || [[ "$content" =~ ^[[:space:]]*\'\'\' ]]; then
                    continue
                fi
                
                # Skip variable names in quotes (string literals)
                if [[ "$content" =~ ['\"].*${pattern}.*['\"] ]]; then
                    continue
                fi
                
                report_violation "$file" "$line_num" "Non-Universal Pattern" \
                    "$(echo "$content" | sed 's/^[[:space:]]*//')" \
                    "Universal RAG should use content-agnostic patterns that work for ANY domain"
            fi
        done < <(grep -n -E "$pattern" "$file" 2>/dev/null || true)
    done
}

# Function to check for example-specific bias in demo scripts
check_demo_bias() {
    local file="$1"
    
    # Skip this check for actual demo/example files that need examples
    if [[ "$file" =~ demo|example|test ]]; then
        return 0
    fi
    
    # Patterns that suggest hardcoded examples or demo-specific content
    local demo_bias_patterns=(
        # Hardcoded example domains in production code
        'example.*technical'
        'demo.*academic'
        'sample.*legal'
        'test.*medical'
        
        # Fixed example content that suggests domain assumptions
        'machine learning.*example'
        'legal.*contract.*example'
        'medical.*diagnosis.*example'
    )
    
    for pattern in "${demo_bias_patterns[@]}"; do
        while IFS=: read -r line_num content; do
            if [[ -n "$content" ]]; then
                # Skip comments
                if [[ "$content" =~ ^[[:space:]]*# ]]; then
                    continue
                fi
                
                report_violation "$file" "$line_num" "Demo/Example Domain Bias" \
                    "$(echo "$content" | sed 's/^[[:space:]]*//')" \
                    "Production code should not contain hardcoded domain-specific examples"
            fi
        done < <(grep -n -E "$pattern" "$file" 2>/dev/null || true)
    done
}

# Main execution
echo -e "Scanning for domain bias violations..."
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
            
            # Run all checks
            check_hardcoded_domains "$file"
            check_config_bias "$file"
            check_domain_classification "$file"
            check_non_universal_patterns "$file"
            check_demo_bias "$file"
            
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
    echo -e "${YELLOW}💡 Universal RAG Principles:${NC}"
    echo "   • DISCOVER domain characteristics from content"
    echo "   • ADAPT parameters based on measured properties"
    echo "   • AVOID predetermined domain categories"
    echo "   • USE content-agnostic patterns and algorithms"
    echo "   • LEARN from data, don't assume domain knowledge"
    echo ""
    echo -e "${YELLOW}🔧 How to fix:${NC}"
    echo "   • Replace hardcoded domain types with discovered characteristics"
    echo "   • Use measured properties (vocabulary_complexity, concept_density) instead of domain labels"
    echo "   • Make configuration adapt to content analysis, not domain assumptions"
    echo "   • Ensure all patterns work universally across ANY domain"
    echo ""
    exit 1
fi