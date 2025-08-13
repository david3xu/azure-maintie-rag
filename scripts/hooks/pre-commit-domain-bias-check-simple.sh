#!/bin/bash
#
# Simple Universal RAG Domain Bias Detection
# ==========================================
# Fast pre-commit check for critical domain bias violations

set -euo pipefail

# Quick domain bias patterns (most critical only)
CRITICAL_PATTERNS=(
    "legal|technical|academic|medical|financial"
    "\.domain\s*==\s*[\"']"
    "hardcode|HARDCODE"
    "\.classify\("
    "domain_type\s*="
)

VIOLATION_COUNT=0

echo "🔍 Quick Domain Bias Check..."

# Check only modified Python files
for pattern in "${CRITICAL_PATTERNS[@]}"; do
    if git diff --cached --name-only | grep -E "\.py$" | xargs grep -l -E "$pattern" 2>/dev/null; then
        echo "❌ Critical domain bias detected: $pattern"
        ((VIOLATION_COUNT++))
    fi
done

if [ $VIOLATION_COUNT -gt 0 ]; then
    echo "❌ Found $VIOLATION_COUNT critical domain bias violations"
    echo "💡 Use discovery patterns instead of hardcoded domain logic"
    exit 1
fi

echo "✅ Quick domain bias check passed"
exit 0