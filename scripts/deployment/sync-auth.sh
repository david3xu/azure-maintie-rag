#!/bin/bash
#
# Azure Authentication Manager for Enterprise Environments
# ========================================================
# 
# Unified script for authentication validation, diagnosis, and pipeline readiness
# Handles university/enterprise Azure AD token expiration issues

set -euo pipefail

# Mode: sync (default) or validate
MODE="${1:-sync}"
PIPELINE_DURATION_MINUTES=20

echo "🔐 Azure Authentication Manager ($MODE mode)"
echo "============================================="

# Check authentication status
check_auth() {
    AZD_STATUS="❌"; CLI_STATUS="❌"
    
    if azd auth login --check-status >/dev/null 2>&1; then
        AZD_USER=$(azd auth login --check-status 2>/dev/null | grep "Logged in" | cut -d' ' -f5- || echo "unknown")
        AZD_STATUS="✅ $AZD_USER"
    fi
    
    if az account show >/dev/null 2>&1; then
        CLI_USER=$(az account show --query user.name -o tsv 2>/dev/null)
        CLI_STATUS="✅ $CLI_USER"
    fi
    
    echo "Status: azd ($AZD_STATUS) | az CLI ($CLI_STATUS)"
}

# Validate subscription consistency
check_subscriptions() {
    AZD_SUB=$(azd env get-values 2>/dev/null | grep "AZURE_SUBSCRIPTION_ID=" | cut -d'=' -f2 | tr -d '"' || echo "")
    CLI_SUB=$(az account show --query id -o tsv 2>/dev/null || echo "")
    
    if [ "$AZD_SUB" = "$CLI_SUB" ] && [ -n "$AZD_SUB" ]; then
        echo "✅ Subscriptions synchronized: $AZD_SUB"
        return 0
    else
        echo "❌ Subscription mismatch: azd($AZD_SUB) vs CLI($CLI_SUB)"
        return 1
    fi
}

# Main logic
check_auth

if [[ "$CLI_STATUS" == "✅"* ]] && [[ "$AZD_STATUS" == "✅"* ]]; then
    check_subscriptions
    SYNC_OK=$?
    
    if [ "$MODE" = "validate" ]; then
        if [ $SYNC_OK -eq 0 ]; then
            echo "🎉 VALIDATION PASSED: Ready for ${PIPELINE_DURATION_MINUTES}-minute pipeline"
            exit 0
        else
            echo "❌ VALIDATION FAILED: Fix subscription mismatch"
            exit 1
        fi
    else
        [ $SYNC_OK -eq 0 ] && echo "🎉 Authentication fully synchronized!" || echo "⚠️  Run: az account set --subscription $AZD_SUB"
    fi
    
elif [[ "$MODE" = "validate" ]]; then
    echo "❌ VALIDATION FAILED: Authentication required"
    echo "🔧 Required: az login && azd auth login"
    exit 1
    
else
    # Diagnosis mode
    if [[ "$AZD_STATUS" == "✅"* ]] && [[ "$CLI_STATUS" == "❌"* ]]; then
        echo "💡 Common issue: Run 'az login' to sync with azd"
    elif [[ "$AZD_STATUS" == "❌"* ]] && [[ "$CLI_STATUS" == "✅"* ]]; then
        echo "💡 Run 'azd auth login' to sync with az CLI"
    else
        echo "💡 Run both: 'azd auth login' && 'az login'"
    fi
    
    echo ""
    echo "🎯 Enterprise token facts:"
    echo "• University/Enterprise AD enforces 1-24h token lifetimes"
    echo "• Pipeline duration: ${PIPELINE_DURATION_MINUTES} minutes"
    echo "• Fresh tokens required before 'make deploy-with-data'"
fi