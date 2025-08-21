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

# Get current azd values
azd env get-values > azd_values.tmp

# If .env exists, preserve it and update/add only new values
if [ -f .env ]; then
    echo "🔄 Updating existing .env file..."
    cp .env .env.backup
    
    # Update existing .env with new values from azd
    while IFS='=' read -r key value; do
        if [ -n "$key" ] && [ -n "$value" ]; then
            # Remove quotes from value if present
            clean_value=$(echo $value | sed 's/^"\(.*\)"$/\1/')
            
            # Update or add the key in .env
            if grep -q "^${key}=" .env; then
                sed -i "s|^${key}=.*|${key}=\"${clean_value}\"|" .env
            else
                echo "${key}=\"${clean_value}\"" >> .env
            fi
        fi
    done < azd_values.tmp
    
    echo "✅ Updated existing .env file with azd values"
else
    echo "📄 Creating new .env file..."
    echo "# Azure Universal RAG Environment Configuration" > .env
    echo "# Generated from azd environment: ${TARGET_ENV}" >> .env
    echo "# $(date)" >> .env
    echo "" >> .env
    cat azd_values.tmp >> .env
    echo "✅ Created new .env file"
fi

# Cleanup
rm -f azd_values.tmp .env.backup

echo ""
echo "🎉 Environment synchronized: $TARGET_ENV"
echo "🚀 Ready to run: make dataflow-full"
