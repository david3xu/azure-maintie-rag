#!/bin/bash
# Simple hardcoded value detection for testing

echo "=== Testing Hardcoded Value Detection ==="
echo

# Test on known files with hardcoded values
TEST_FILES=(
    "agents/domain_intelligence/analyzers/hybrid_configuration_generator.py"
    "agents/domain_intelligence/analyzers/unified_content_analyzer.py"
    "agents/core/cache_manager.py"
)

for file in "${TEST_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "🔍 Checking $file"
        # Look for decimal values 0.5-0.9
        grep -n "0\.[5-9][0-9]*" "$file" | grep -v "constants\." | grep -v "#" | head -3
        echo
    fi
done

echo "=== Total count of hardcoded values in agents/ ==="
find agents/ -name "*.py" -exec grep -l "0\.[5-9][0-9]*" {} \; | grep -v constants.py | wc -l