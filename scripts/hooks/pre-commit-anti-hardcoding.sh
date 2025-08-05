#!/bin/bash
# scripts/hooks/pre-commit-anti-hardcoding.sh
# Pre-commit hook to detect and block hardcoded configuration values

echo "🔍 Checking for hardcoded values..."

# Check for hardcoded configuration patterns
HARDCODED_PATTERNS=(
    "similarity_threshold\s*=\s*[0-9]" 
    "processing_delay\s*=\s*[0-9]"
    "synthesis_weight\s*=\s*[0-9]"
    "mock_implementation"
    "TODO.*config"
    "hardcoded"
    "placeholder"
)

VIOLATIONS_FOUND=0

for pattern in "${HARDCODED_PATTERNS[@]}"; do
    if git diff --cached --name-only | xargs grep -l "$pattern" 2>/dev/null; then
        echo "❌ HARDCODED VALUE DETECTED: $pattern"
        git diff --cached --name-only | xargs grep -n "$pattern"
        VIOLATIONS_FOUND=1
    fi
done

# Check for NotImplementedError in config classes
if git diff --cached --name-only | xargs grep -l "raise NotImplementedError" agents/core/ 2>/dev/null; then
    echo "❌ CONFIGURATION CLASSES WITH NotImplementedError:"
    git diff --cached --name-only | xargs grep -n "raise NotImplementedError" agents/core/
    VIOLATIONS_FOUND=1
fi

if [ $VIOLATIONS_FOUND -eq 1 ]; then
    echo ""
    echo "🚨 COMMIT BLOCKED: Hardcoded values detected!"
    echo "   The system is designed for intelligent, data-driven configuration."
    echo "   Please implement proper Config-Extraction → Search workflow integration."
    echo ""
    echo "   To bypass for development: git commit --no-verify"
    exit 1
fi

echo "✅ No hardcoded values detected. Commit allowed."