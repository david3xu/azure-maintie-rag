#!/bin/bash
# Auto-sync azd environment to backend configuration
# Usage: ./scripts/sync-env.sh [environment]
# If no environment specified, uses current azd environment

set -e

# Determine target environment
if [ -n "$1" ]; then
    TARGET_ENV="$1"
    echo "🎯 Using specified environment: $TARGET_ENV"
    # Switch azd to target environment
    azd env select "$TARGET_ENV"
else
    # Get current azd environment
    TARGET_ENV=$(azd env get-values 2>/dev/null | grep "AZURE_ENV_NAME=" | cut -d'=' -f2 | tr -d '"' || echo "")
    if [ -z "$TARGET_ENV" ]; then
        echo "❌ No azd environment selected and none specified"
        echo "Usage: $0 [environment]"
        echo "Available: azd env list"
        exit 1
    fi
    echo "🔄 Using current azd environment: $TARGET_ENV"
fi

echo "📁 Syncing backend configuration..."

# Create environment file from current azd values
azd env get-values > "backend/config/environments/${TARGET_ENV}.env"
echo "✅ Created: backend/config/environments/${TARGET_ENV}.env"

# Update .env symlink
cd backend
rm -f .env
ln -sf "config/environments/${TARGET_ENV}.env" .env
echo "✅ Updated: .env -> config/environments/${TARGET_ENV}.env"

# Update Makefile default environment
cd ..
sed -i "s/AZURE_ENVIRONMENT := .*/AZURE_ENVIRONMENT := \$(or \$(AZURE_ENVIRONMENT), ${TARGET_ENV})/" backend/Makefile
echo "✅ Updated: Makefile default environment"

echo ""
echo "🎉 Backend synchronized with azd environment: $TARGET_ENV"
echo "🚀 Ready to run: make setup && make run"