#!/bin/bash

echo "🔍 DEBUGGING AZURE DEPLOYMENT STATUS"
echo "===================================="
echo

echo "1️⃣ Checking active processes..."
echo "Active make/azd processes:"
ps aux | grep -E "(make|azd)" | grep -v grep || echo "   No active deployment processes found"
echo

echo "2️⃣ Checking Azure authentication..."
az account show --query "{user:user.name,subscription:name,tenantId:tenantId}" -o table 2>/dev/null || echo "   Authentication issue detected"
echo

echo "3️⃣ Checking deployed resources status..."
echo "Container Apps:"
az containerapp list --resource-group rg-maintie-rag-prod --query "[].{name:name,status:properties.runningStatus,fqdn:properties.configuration.ingress.fqdn}" -o table 2>/dev/null || echo "   Could not check Container Apps"
echo

echo "4️⃣ Checking Azure AI Foundry status..."
echo "Cognitive Services (AI Foundry):"
az cognitiveservices account list --resource-group rg-maintie-rag-prod --query "[].{name:name,kind:kind,provisioningState:properties.provisioningState}" -o table 2>/dev/null || echo "   Could not check AI services"
echo

echo "5️⃣ Checking recent deployments..."
echo "Recent Azure deployments:"
az deployment group list --resource-group rg-maintie-rag-prod --query "[0:3].{name:name,state:properties.provisioningState,timestamp:properties.timestamp}" -o table 2>/dev/null || echo "   Could not check deployments"
echo

echo "6️⃣ Checking if backend is responding..."
echo "Testing backend health:"
timeout 5 curl -s "https://ca-backend-maintie-rag-prod.redplant-df598dc7.westus2.azurecontainerapps.io/health" > /dev/null && echo "   ✅ Backend responding" || echo "   ❌ Backend not responding"
echo

echo "7️⃣ Quick Azure AI test..."
echo "Testing Azure AI Foundry API:"
timeout 5 curl -s -H "Authorization: Bearer $(az account get-access-token --resource https://cognitiveservices.azure.com --query accessToken -o tsv 2>/dev/null)" \
    "https://29192-medxawl1-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2024-06-01" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"test"}],"max_tokens":5}' 2>/dev/null | jq -r '.choices[0].message.content' 2>/dev/null && echo "   ✅ Azure AI Foundry working" || echo "   ❌ Azure AI Foundry not responding"

echo
echo "🎯 DIAGNOSIS COMPLETE"
echo "Check above for any failed components."