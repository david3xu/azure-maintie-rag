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

echo -e "${BLUE}🔍 Universal RAG Domain Bias & Mock Value Detection${NC}"
echo "===================================================="

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
    
    # Domain category patterns that violate universal philosophy - COMPREHENSIVE LIST
    local domain_patterns=(
        # Predetermined domain types (significantly expanded)
        '"technical"'
        '"academic"'
        '"legal"'
        '"medical"'
        '"financial"'
        '"scientific"'
        '"business"'
        '"research"'
        '"educational"'
        '"engineering"'
        '"healthcare"'
        '"pharmaceutical"'
        '"automotive"'
        '"aerospace"'
        '"manufacturing"'
        '"industrial"'
        '"corporate"'
        '"administrative"'
        '"clinical"'
        '"biomedical"'
        '"software"'
        '"programming"'
        '"coding"'
        '"development"'
        '"maintenance"'
        '"tutorial"'
        '"documentation"'
        '"manual"'
        '"guide"'
        '"handbook"'
        '"procedure"'
        '"protocol"'
        '"specification"'
        '"standard"'
        '"regulatory"'
        '"compliance"'
        '"policy"'
        '"government"'
        '"public"'
        '"private"'
        '"commercial"'
        '"enterprise"'
        '"consumer"'
        "'technical'"
        "'academic'"
        "'legal'"
        "'medical'"
        "'financial'"
        "'scientific'"
        "'business'"
        "'research'"
        "'educational'"
        "'engineering'"
        "'healthcare'"
        "'programming'"
        "'maintenance'"
        "'industrial'"
        "'software'"
        "'clinical'"
        
        # Domain classification assumptions (simplified)
        'preset_domains'
        'known_domains' 
        'supported_domains'
        'PREDEFINED_DOMAINS'
        
        # Specific domain return patterns that violate Universal RAG
        'return.*"programming"'
        'return.*"medical_research"'
        'return.*"industrial_maintenance"'
        'return.*"python_programming"'
        'return.*"technical_documentation"'
        'return.*"academic_paper"'
        'return.*"legal_document"'
        'return.*"clinical_study"'
        'return.*"business_document"'
        'return.*"software_development"'
        'return.*"engineering_manual"'
        'return.*"healthcare_protocol"'
        '"programming_tutorial"'
        '"medical_research"'
        '"maintenance_procedures"'
        '"technical_manual"'
        '"academic_content"'
        '"legal_content"'
        '"clinical_data"'
        '"business_content"'
        '"software_documentation"'
        '"engineering_specification"'
        '"healthcare_documentation"'
        '"financial_report"'
        '"scientific_paper"'
        '"research_document"'
        
        # PRODUCTION QUALITY VIOLATIONS: Fallback/Mock patterns that hide real issues (simplified)
        'fallback.*simulation'
        'simulated.*data'
        'mock.*data'
        'fake.*data'
        'return.*simulated'
        'return.*mock'
        
        # Entity type assumptions that violate Universal RAG
        '"TECHNICAL_TERM"'
        '"MEDICAL_TERM"'
        '"LEGAL_ENTITY"'
        '"ACADEMIC_CONCEPT"'
        '"BUSINESS_ENTITY"'
        '"PROGRAMMING_CONCEPT"'
        '"SOFTWARE_COMPONENT"'
        '"CLINICAL_TERM"'
        '"FINANCIAL_INSTRUMENT"'
        '"SCIENTIFIC_CONCEPT"'
        '"ENGINEERING_COMPONENT"'
        '"HEALTHCARE_TERM"'
        '"INDUSTRIAL_COMPONENT"'
        '"RESEARCH_METHOD"'
        '"EDUCATIONAL_CONCEPT"'
        
        # Vocabulary assumptions that violate Universal RAG
        'technical_vocabulary'
        'medical_vocabulary'
        'legal_vocabulary'
        'academic_vocabulary'
        'business_vocabulary'
        'programming_vocabulary'
        'software_vocabulary'
        'clinical_vocabulary'
        'financial_vocabulary'
        'scientific_vocabulary'
        'engineering_vocabulary'
        'healthcare_vocabulary'
        'industrial_vocabulary'
        'domain_vocabulary'
        'specialized_vocabulary'
        'technical_terms'
        'medical_terms'
        'legal_terms'
        'academic_terms'
        'business_terms'
        'programming_terms'
        'software_terms'
        'clinical_terms'
        'financial_terms'
        'scientific_terms'
        'engineering_terms'
        'healthcare_terms'
        'industrial_terms'
        'expected_entity_types'
        'predefined_entities'
        'known_entity_types'
        'domain_entities'
        'specialized_entities'
        'technical_entities'
        'medical_entities'
        'legal_entities'
        'academic_entities'
        'business_entities'
        
        # Processing assumptions based on domains that violate Universal RAG
        'Domain specific'
        'domain_specific'
        
        # NEWLY DETECTED: Complexity assumptions that violate Universal RAG
        '"simple".*query'
        '"complex".*query'
        '"basic".*search'
        '"advanced".*search'
        '"standard".*analysis'
        '"premium".*processing'
        '"enterprise".*mode'
        '"professional".*tier'
        '"beginner".*level'
        '"intermediate".*level'
        '"expert".*level'
        '"low".*complexity'
        '"medium".*complexity'
        '"high".*complexity'
        '"simple".*algorithm'
        '"complex".*algorithm'
        
        # NEWLY DETECTED: Quality tier assumptions
        '"basic".*quality'
        '"standard".*quality'
        '"premium".*quality'
        '"enterprise".*quality'
        'quality.*tier'
        'processing.*tier'
        'service.*tier'
        'performance.*tier'
        'complexity.*tier'
        
        # NEWLY DETECTED: User type assumptions
        '"consumer".*type'
        '"enterprise".*type'
        '"professional".*type'
        '"commercial".*type'
        '"personal".*type'
        'user_type.*=='
        'client_type.*=='
        
        # NEWLY DETECTED: Content difficulty assumptions
        'difficulty.*level'
        'reading.*level'
        'comprehension.*level'
        'skill.*level'
        'expertise.*level'
        'proficiency.*level'
    )
    
    for pattern in "${domain_patterns[@]}"; do
        while IFS=: read -r line_num content; do
            if [[ -n "$content" ]]; then
                # Skip comments and documentation
                if [[ "$content" =~ ^[[:space:]]*# ]] || [[ "$content" =~ ^[[:space:]]*\"\"\" ]] || [[ "$content" =~ ^[[:space:]]*\'\'\' ]]; then
                    continue
                fi
                
                # Skip legitimate infrastructure validation patterns
                if [[ "$content" =~ "if not azure_endpoint" ]] || [[ "$content" =~ "if not validation\[" ]] || [[ "$content" =~ "if not _azure_credential" ]]; then
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
    
    # Configuration patterns that violate domain-agnostic principles - COMPREHENSIVE 
    local config_bias_patterns=(
        # Domain-based threshold assumptions (expanded)
        'technical_vocabulary_ratio.*>.*[0-9]'
        'medical_vocabulary_ratio.*>.*[0-9]'
        'legal_vocabulary_ratio.*>.*[0-9]'
        'academic_vocabulary_ratio.*>.*[0-9]'
        'business_vocabulary_ratio.*>.*[0-9]'
        'programming_vocabulary_ratio.*>.*[0-9]'
        'scientific_vocabulary_ratio.*>.*[0-9]'
        'engineering_vocabulary_ratio.*>.*[0-9]'
        'if.*technical.*density'
        'if.*medical.*density'
        'if.*legal.*density'
        'if.*academic.*density'
        'if.*business.*density'
        'if.*programming.*density'
        'if.*scientific.*density'
        'if.*engineering.*density'
        'technical.*content'
        'medical.*content'
        'legal.*content'
        'academic.*document'
        'business.*document'
        'programming.*document'
        'scientific.*document'
        'engineering.*document'
        'clinical.*document'
        'financial.*document'
        'industrial.*document'
        'software.*document'
        'healthcare.*document'
        'legal.*terminology'
        'medical.*terminology'
        'technical.*terminology'
        'academic.*terminology'
        'business.*terminology'
        'programming.*terminology'
        'scientific.*terminology'
        'engineering.*terminology'
        'clinical.*terminology'
        'financial.*terminology'
        'industrial.*terminology'
        'software.*terminology'
        'healthcare.*terminology'
        
        # Hardcoded threshold adjustments based on domain assumptions (simplified)
        'if.*domain_specific.*>'
        'technical_indicators.*='
        'medical_indicators.*='
        'legal_indicators.*='
        'academic_indicators.*='
        'business_indicators.*='
        'programming_indicators.*='
        'scientific_indicators.*='
        'engineering_indicators.*='
        'clinical_indicators.*='
        'financial_indicators.*='
        'industrial_indicators.*='
        'software_indicators.*='
        'healthcare_indicators.*='
        'domain_indicators.*='
        'specialized_indicators.*='
        
        # Predetermined entity types (expanded)
        'expected_entity_types.*=.*\[.*["'"'"']CONCEPT["'"'"']'
        'expected_entity_types.*=.*\[.*["'"'"']TERM["'"'"']'
        'expected_entity_types.*=.*\[.*["'"'"']PROCEDURE["'"'"']'
        'expected_entity_types.*=.*\[.*["'"'"']ENTITY["'"'"']'
        'technical_words.*=.*\['
        'medical_words.*=.*\['
        'legal_words.*=.*\['
        'academic_terms.*=.*\['
        'business_terms.*=.*\['
        'programming_terms.*=.*\['
        'scientific_terms.*=.*\['
        'engineering_terms.*=.*\['
        'clinical_terms.*=.*\['
        'financial_terms.*=.*\['
        'industrial_terms.*=.*\['
        'software_terms.*=.*\['
        'healthcare_terms.*=.*\['
        'domain_terms.*=.*\['
        'specialized_terms.*=.*\['
        'predefined_terms.*=.*\['
        'known_terms.*=.*\['
        'preset_terms.*=.*\['
        'default_terms.*=.*\['
        
        # Configuration mappings that assume domains
        'domain_config_map'
        'domain_settings_map'
        'domain_threshold_map'
        'domain_parameter_map'
        
        # Hardcoded processing parameters based on domain assumptions (simplified)
        'domain_chunk_size'
        'domain_overlap'
        'domain_threshold'
        
        # NEWLY DETECTED: Implicit complexity bias patterns
        'if.*simple.*query'
        'if.*complex.*content'
        'if.*basic.*document'
        'if.*advanced.*processing'
        'simple.*vs.*complex'
        'basic.*vs.*advanced'
        'low.*vs.*high.*complexity'
        'beginner.*vs.*expert'
        'consumer.*vs.*enterprise'
        'personal.*vs.*professional'
        
        # NEWLY DETECTED: Service tier bias
        'tier.*==.*"basic"'
        'tier.*==.*"standard"'
        'tier.*==.*"premium"'
        'tier.*==.*"enterprise"'
        'sku.*==.*"basic"'
        'sku.*==.*"standard"'
        'sku.*==.*"premium"'
        
        # NEWLY DETECTED: Processing complexity assumptions
        'processing_complexity.*==.*"low"'
        'processing_complexity.*==.*"medium"'
        'processing_complexity.*==.*"high"'
        'algorithm_complexity.*='
        'computational_complexity.*='
        'query_complexity.*weights'
        
        # NEWLY DETECTED: User experience assumptions
        'user_experience.*level'
        'expertise_required'
        'skill_level_required'
        'difficulty_rating'
        'complexity_rating'
        'proficiency_needed'
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
    
    # Patterns that classify content into predetermined categories - COMPREHENSIVE
    local classification_patterns=(
        # Domain classification functions (expanded)
        'classify_domain'
        'classify_content_type'
        'classify_document_type'
        'detect_domain_type'
        'detect_content_type'
        'detect_document_type'
        'determine_domain'
        'determine_content_type'
        'determine_document_type'
        'identify_domain_category'
        'identify_content_category'
        'identify_document_category'
        'analyze_domain_type'
        'analyze_content_type'
        'analyze_document_type'
        'categorize_domain'
        'categorize_content'
        'categorize_document'
        'predict_domain'
        'predict_content_type'
        'predict_document_type'
        'infer_domain'
        'infer_content_type'
        'infer_document_type'
        'recognize_domain'
        'recognize_content_type'
        'recognize_document_type'
        'extract_domain_type'
        'extract_content_type'
        'extract_document_type'
        
        # If-else chains for domain types (expanded)
        'elif.*domain.*==.*["'"'"']'
        'elif.*content.*==.*["'"'"']'
        'elif.*document.*==.*["'"'"']'
        'if.*domain_type.*in'
        'if.*content_type.*in'
        'if.*document_type.*in'
        'domain_type.*in.*\['
        'content_type.*in.*\['
        'document_type.*in.*\['
        'domain.*==.*["'"'"']technical'
        'domain.*==.*["'"'"']medical'
        'domain.*==.*["'"'"']legal'
        'domain.*==.*["'"'"']academic'
        'domain.*==.*["'"'"']business'
        'domain.*==.*["'"'"']programming'
        'domain.*==.*["'"'"']scientific'
        'domain.*==.*["'"'"']engineering'
        'content.*==.*["'"'"']technical'
        'content.*==.*["'"'"']medical'
        'content.*==.*["'"'"']legal'
        'content.*==.*["'"'"']academic'
        'content.*==.*["'"'"']business'
        'content.*==.*["'"'"']programming'
        'content.*==.*["'"'"']scientific'
        'content.*==.*["'"'"']engineering'
        
        # Domain-specific processing branches (expanded)
        'if.*is_technical'
        'if.*is_medical'
        'if.*is_legal'
        'if.*is_academic'
        'if.*is_business'
        'if.*is_programming'
        'if.*is_scientific'
        'if.*is_engineering'
        'if.*is_clinical'
        'if.*is_financial'
        'if.*is_industrial'
        'if.*is_software'
        'if.*is_healthcare'
        'process_technical'
        'process_medical'
        'process_legal'
        'process_academic'
        'process_business'
        'process_programming'
        'process_scientific'
        'process_engineering'
        'process_clinical'
        'process_financial'
        'process_industrial'
        'process_software'
        'process_healthcare'
        'handle_technical'
        'handle_medical'
        'handle_legal'
        'handle_academic'
        'handle_business'
        'handle_programming'
        'handle_scientific'
        'handle_engineering'
        'handle_clinical'
        'handle_financial'
        'handle_industrial'
        'handle_software'
        'handle_healthcare'
        'handle_domain_type'
        'handle_content_type'
        'handle_document_type'
        
        # Switch statements and case patterns for domains
        'case.*technical'
        'case.*medical'
        'case.*legal'
        'case.*academic'
        'case.*business'
        'case.*programming'
        'case.*scientific'
        'case.*engineering'
        'switch.*domain'
        'switch.*content'
        'switch.*document'
        
        # Domain mapping and routing
        'route_by_domain'
        'route_by_content_type'
        'route_by_document_type'
        'dispatch_by_domain'
        'dispatch_by_content_type'
        'dispatch_by_document_type'
        'map_domain_to'
        'map_content_to'
        'map_document_to'
        'select_by_domain'
        'select_by_content_type'
        'select_by_document_type'
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
    
    # Only skip this check for pure test files, not demos that might violate Universal RAG
    if [[ "$file" =~ test.*\.py$ ]] && [[ "$file" =~ tests/ ]]; then
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
# Function to check for mock/fake values (production quality enforcement)
check_mock_values() {
    local file="$1"
    
    # Mock value patterns that should never appear in production code
    local mock_patterns=(
        # Mock API keys and endpoints
        'sk-mock'
        'mock-key'
        'fake-key'
        'test-key'
        'dummy-key'
        'mock.*endpoint'
        'fake.*endpoint'
        'test.*endpoint'
        'dummy.*endpoint'
        'localhost.*azure'
        'mock.*azure'
        'fake.*azure'
        'test.*azure'
        'dummy.*azure'
        # Mock Azure resources
        'mock.*storage'
        'fake.*storage' 
        'test.*storage'
        'dummy.*storage'
        'mock.*cosmos'
        'fake.*cosmos'
        'test.*cosmos'
        'dummy.*cosmos'
        'mock.*openai'
        'fake.*openai'
        'test.*openai'
        'dummy.*openai'
        # Mock credentials
        'mock.*credential'
        'fake.*credential'
        'test.*credential'
        'dummy.*credential'
        'placeholder.*key'
        'placeholder.*endpoint'
        'placeholder.*url'
        # Environment placeholders
        'YOUR_.*_HERE'
        'REPLACE_.*_VALUE'
        'INSERT_.*_HERE'
        'ADD_YOUR_.*'
        'CHANGE_THIS_.*'
    )
    
    for pattern in "${mock_patterns[@]}"; do
        while IFS=: read -r line_num content; do
            if [[ -n "$content" ]]; then
                # Skip comments and documentation examples
                if [[ "$content" =~ ^[[:space:]]*# ]] && [[ "$content" =~ (example|sample|demo) ]]; then
                    continue
                fi
                
                report_violation "$file" "$line_num" "Mock/Fake Value Detected" \
                    "$(echo "$content" | sed 's/^[[:space:]]*//')" \
                    "Production code must use real Azure services only - no mocks/fakes allowed"
            fi
        done < <(grep -n -i -E "$pattern" "$file" 2>/dev/null || true)
    done
}
