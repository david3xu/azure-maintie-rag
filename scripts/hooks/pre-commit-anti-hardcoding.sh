#!/bin/bash
# Pre-commit hook to enforce zero-hardcoded-values and centralized data types in agents/ directory
# Detects and reports:
# 1. Hardcoded values that should be moved to agents/core/constants.py
# 2. Mathematical expressions that should be moved to agents/core/math_expressions.py  
# 3. Scattered data model definitions that should be moved to agents/core/data_models.py

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGENTS_DIR="${PROJECT_ROOT}/agents"

# Define directories to check
CHECK_DIRS=("${PROJECT_ROOT}/agents" "${PROJECT_ROOT}/api" "${PROJECT_ROOT}/services" "${PROJECT_ROOT}/infrastructure")
EXISTING_DIRS=()

# Check which directories exist
for dir in "${CHECK_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        EXISTING_DIRS+=("$dir")
    fi
done

if [[ ${#EXISTING_DIRS[@]} -eq 0 ]]; then
    echo -e "${RED}❌ Error: No target directories found (agents/, api/, services/, infrastructure/)${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Checking directories for hardcoded values and centralized data models...${NC}"
echo -e "${BLUE}Directories: ${EXISTING_DIRS[*]}${NC}"
echo

# Find all Python files in agents directory (excluding allowed files)
violations=0
files_checked=0

# Allowed files for hardcoded values
ALLOWED_FILES=("constants.py" "math_expressions.py" "data_models.py" "__init__.py")

# Iterate through all existing directories
for check_dir in "${EXISTING_DIRS[@]}"; do
    echo -e "${BLUE}📂 Checking directory: $check_dir${NC}"
    
    while IFS= read -r -d '' file; do
        filename=$(basename "$file")
        relative_path="${file#$PROJECT_ROOT/}"
        
        # Skip allowed files only in agents/core/
        skip_file=false
        if [[ "$file" == *"/agents/core/"* ]]; then
            for allowed in "${ALLOWED_FILES[@]}"; do
                if [[ "$filename" == "$allowed" ]]; then
                    skip_file=true
                    break
                fi
            done
        fi
        
        if [[ "$skip_file" == true ]]; then
            continue
        fi
        
        files_checked=$((files_checked + 1))
        
        # Check for hardcoded values: decimals (0.1-0.9), integers (1000+), and common thresholds
        # Exclude lines that reference constants.py, math_expressions.py, comments, docstrings, and parameter documentation
        hardcoded_lines=$(grep -n -E "(0\.[0-9]+|[0-9]{3,}|confidence=0\.[0-9]+|threshold.*[0-9]+)" "$file" 2>/dev/null | grep -v -E "(constants\.|math_expressions\.|MATH\.|EXPR\.|#|\"\"\"|'''|Args:|[0-9]+:\s*[a-zA-Z_][a-zA-Z0-9_]*:|\[:)" || true)
        
        if [[ -n "$hardcoded_lines" ]]; then
            echo -e "${RED}❌ Hardcoded values found in $relative_path:${NC}"
            echo "$hardcoded_lines" | while IFS= read -r line; do
                echo -e "${YELLOW}  $line${NC}"
                violations=$((violations + 1))
            done
            echo
        fi
        
    done < <(find "$check_dir" -name "*.py" -type f -print0)
done

echo -e "${BLUE}📊 Summary:${NC}"
echo "  Files checked: $files_checked"

# Count total violations properly (excluding allowed files and references to constants/math_expressions)
violation_files=()
for check_dir in "${EXISTING_DIRS[@]}"; do
    while IFS= read -r -d '' file; do
        filename=$(basename "$file")
        
        # Skip allowed files only in agents/core/
        skip_file=false
        if [[ "$file" == *"/agents/core/"* ]]; then
            for allowed in "${ALLOWED_FILES[@]}"; do
                if [[ "$filename" == "$allowed" ]]; then
                    skip_file=true
                    break
                fi
            done
        fi
        
        if [[ "$skip_file" == true ]]; then
            continue
        fi
        
        # Check if file has hardcoded values (excluding references to constants/math, comments, and parameter docs)
        violations_found=$(grep -n -E "(0\.[0-9]+|[0-9]{3,}|confidence=0\.[0-9]+|threshold.*[0-9]+)" "$file" 2>/dev/null | grep -v -E "(constants\.|math_expressions\.|MATH\.|EXPR\.|#|\"\"\"|'''|Args:|[0-9]+:\s*[a-zA-Z_][a-zA-Z0-9_]*:|\[:)" || true)
        if [[ -n "$violations_found" ]]; then
            violation_files+=("$file")
        fi
    done < <(find "$check_dir" -name "*.py" -type f -print0)
done

total_violations=${#violation_files[@]}

echo
echo -e "${BLUE}🎯 Checking for centralized data types...${NC}"

# Check for scattered data model definitions - different rules for agents/ vs other directories
scattered_models_agents=()
scattered_models_others=()
data_types_checked_agents=0
data_types_checked_others=0

echo -e "${BLUE}🎯 Checking agents/ for centralized data types (strict enforcement)...${NC}"
# Search for BaseModel, dataclass, and Enum definitions in agents/ outside core/data_models.py
if [[ -d "${PROJECT_ROOT}/agents" ]]; then
    while IFS= read -r -d '' file; do
        filename=$(basename "$file")
        relative_path="${file#$PROJECT_ROOT/}"
        
        # Skip the centralized data models file and __init__.py files
        if [[ "$filename" == "data_models.py" || "$filename" == "__init__.py" ]]; then
            continue
        fi
        
        data_types_checked_agents=$((data_types_checked_agents + 1))
        
        # Check for data model definitions
        model_definitions=$(grep -n -E "(class.*BaseModel|class.*TypedDict|@dataclass)" "$file" 2>/dev/null || true)
        
        if [[ -n "$model_definitions" ]]; then
            scattered_models_agents+=("$file")
            echo -e "${RED}❌ Scattered data model definitions found in $relative_path:${NC}"
            echo "$model_definitions" | while IFS= read -r line; do
                echo -e "${YELLOW}  $line${NC}"
            done
            echo -e "${YELLOW}💡 Move to agents/core/data_models.py${NC}"
            echo
        fi
    done < <(find "${PROJECT_ROOT}/agents" -name "*.py" -not -path "*/core/data_models.py" -type f -print0)
fi

echo -e "${BLUE}🔍 Checking other directories for data model imports (recommendation)...${NC}"
# Check other directories for data models that should potentially import from agents/core/data_models.py
for check_dir in "${EXISTING_DIRS[@]}"; do
    if [[ "$check_dir" == "${PROJECT_ROOT}/agents" ]]; then
        continue  # Skip agents, already checked above
    fi
    
    if [[ ! -d "$check_dir" ]]; then
        continue
    fi
    
    echo -e "${BLUE}📂 Checking directory: $check_dir${NC}"
    
    while IFS= read -r -d '' file; do
        filename=$(basename "$file")
        relative_path="${file#$PROJECT_ROOT/}"
        
        # Skip __init__.py files
        if [[ "$filename" == "__init__.py" ]]; then
            continue
        fi
        
        data_types_checked_others=$((data_types_checked_others + 1))
        
        # Check for data model definitions
        model_definitions=$(grep -n -E "(class.*BaseModel|class.*TypedDict|@dataclass)" "$file" 2>/dev/null || true)
        
        # Check if they import from agents.core.data_models
        imports_centralized=$(grep -n -E "from agents\.core\.data_models import|import agents\.core\.data_models" "$file" 2>/dev/null || true)
        
        if [[ -n "$model_definitions" ]]; then
            if [[ -z "$imports_centralized" ]]; then
                scattered_models_others+=("$file")
                echo -e "${YELLOW}⚠️ Data models found in $relative_path without centralized import:${NC}"
                echo "$model_definitions" | while IFS= read -r line; do
                    echo -e "${YELLOW}  $line${NC}"
                done
                echo -e "${BLUE}💡 Consider importing from agents.core.data_models if applicable${NC}"
                echo
            fi
        fi
    done < <(find "$check_dir" -name "*.py" -type f -print0)
done

scattered_model_count_agents=${#scattered_models_agents[@]}
scattered_model_count_others=${#scattered_models_others[@]}

echo -e "${BLUE}📊 Data Types Summary:${NC}"
echo "  Agents files checked: $data_types_checked_agents"
echo "  Other files checked: $data_types_checked_others"
echo "  Agents files with scattered models (strict): $scattered_model_count_agents"
echo "  Other files without centralized imports (warning): $scattered_model_count_others"
echo

# Final validation results
if [[ $total_violations -eq 0 && $scattered_model_count_agents -eq 0 ]]; then
    echo -e "${GREEN}🎉 ZERO-HARDCODED-VALUES TRINITY COMPLETE! 🎉${NC}"
    echo -e "${GREEN}✅ Values: All constants properly located in agents/core/constants.py${NC}"
    echo -e "${GREEN}✅ Calculations: All math expressions properly located in agents/core/math_expressions.py${NC}"
    echo -e "${GREEN}✅ Types: All agent data models properly located in agents/core/data_models.py${NC}"
    
    if [[ $scattered_model_count_others -gt 0 ]]; then
        echo -e "${YELLOW}⚠️ Recommendations for other directories:${NC}"
        echo -e "${BLUE}Consider importing from agents.core.data_models when appropriate${NC}"
    fi
    
    exit 0
elif [[ $total_violations -eq 0 && $scattered_model_count_agents -gt 0 ]]; then
    echo -e "${YELLOW}⚠️ Hardcoded values: PASSED${NC}"
    echo -e "${RED}❌ Agent data types centralization: FAILED${NC}"
    echo -e "${YELLOW}💡 Move all agent Pydantic models, dataclasses, and TypedDicts to agents/core/data_models.py${NC}"
    echo
    echo -e "${BLUE}Agent files with scattered data models:${NC}"
    for file in "${scattered_models_agents[@]}"; do
        echo "  - $file"
    done
    
    if [[ $scattered_model_count_others -gt 0 ]]; then
        echo -e "${YELLOW}📝 Also consider centralized imports in other directories${NC}"
    fi
    
    exit 1
elif [[ $total_violations -gt 0 && $scattered_model_count_agents -eq 0 ]]; then
    echo -e "${RED}❌ Hardcoded values: FAILED${NC}"
    echo -e "${GREEN}✅ Agent data types centralization: PASSED${NC}"
    echo -e "${YELLOW}💡 Move constants to agents/core/constants.py and mathematical expressions to agents/core/math_expressions.py${NC}"
    echo
    echo -e "${BLUE}Files with hardcoded values:${NC}"
    for file in "${violation_files[@]}"; do
        echo "  - $file"
    done
    exit 1
else
    echo -e "${RED}❌ Hardcoded values: FAILED (${total_violations} files)${NC}"
    echo -e "${RED}❌ Agent data types centralization: FAILED (${scattered_model_count_agents} files)${NC}"
    echo
    echo -e "${YELLOW}💡 Actions needed:${NC}"
    echo -e "${YELLOW}1. Move constants to agents/core/constants.py${NC}"
    echo -e "${YELLOW}2. Move mathematical expressions to agents/core/math_expressions.py${NC}"
    echo -e "${YELLOW}3. Move all agent data models to agents/core/data_models.py${NC}"
    echo
    echo -e "${BLUE}Files with hardcoded values:${NC}"
    for file in "${violation_files[@]}"; do
        echo "  - $file"
    done
    echo
    echo -e "${BLUE}Agent files with scattered data models:${NC}"
    for file in "${scattered_models_agents[@]}"; do
        echo "  - $file"
    done
    
    if [[ $scattered_model_count_others -gt 0 ]]; then
        echo
        echo -e "${YELLOW}Other files without centralized imports (recommendations):${NC}"
        for file in "${scattered_models_others[@]}"; do
            echo "  - $file"
        done
    fi
    
    exit 1
fi