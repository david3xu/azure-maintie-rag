# Azure Universal RAG Infrastructure

Production-ready Azure infrastructure using Bicep templates.

## 🚨 CRITICAL NOTES

**Fixed and stable architecture:**
- ✅ All naming conflicts resolved
- ✅ Consistent `uniqueString()` patterns 
- ✅ Cost-optimized for Azure for Students
- ✅ No resource overlaps

**⚠️ KEY RULES:**
1. All resources use: `uniqueString(resourceGroup().id, resourcePrefix, environmentName)`
2. Test changes in non-production first
3. No resource overlaps between modules

## 📁 Architecture

```
infra/
├── main.bicep                    # Main entry point
└── modules/
    ├── core-services.bicep       # Storage, Search, KeyVault, Monitoring, Identity  
    ├── ai-services.bicep         # Azure OpenAI (gpt-4.1-mini, embeddings)
    ├── data-services.bicep       # Cosmos DB Gremlin, Azure ML Workspace
    └── hosting-services.bicep    # Container Apps, Container Registry
```

**Dependencies:** core-services → data-services, hosting-services

## 📋 Key Resources

| Service | Purpose | Cost Optimization |
|---------|---------|-------------------|
| **Azure OpenAI** | gpt-4.1-mini, embeddings | Minimal capacity units |
| **Cognitive Search** | Vector search | BASIC tier (avoids quota conflicts) |
| **Cosmos DB** | Graph database (Gremlin) | Serverless (1M RU/s + 25GB free) |
| **Storage Account** | Document storage | Standard_LRS, Cool tier |
| **Container Apps** | API/UI hosting | Scale-to-zero |
| **Azure ML** | GNN training | CPU-only compute |

## 🎯 Naming Pattern

**All resources use this stable pattern:**
```bicep
name: 'prefix-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id, resourcePrefix, environmentName)}'
```

## ⚠️ CRITICAL: Azure for Students Working Configuration

**🚨 DO NOT CHANGE THESE VALUES - DEPLOYMENT WILL FAIL WITHOUT THESE EXACT SETTINGS**

### **Root Cause Analysis: Why Azure for Students is Different**

**Azure for Students subscriptions have STRICT quota limitations:**
1. **Limited AI service types** - Only specific kinds are allowed
2. **Restricted model versions** - Not all OpenAI model versions are available  
3. **Geographic restrictions** - Limited regions have adequate quota
4. **SKU limitations** - Standard enterprise SKUs are blocked
5. **Special approval required** - Some services need pre-approval (which students can't get)

### **🔧 EXACT WORKING CONFIGURATION (tested and verified)**

```bicep
// File: infra/modules/ai-services.bicep
// These values were determined through systematic testing and Azure REST API validation

// 1. COGNITIVE SERVICES ACCOUNT
resource aiServicesAccount 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name: 'oai-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id, resourcePrefix, environmentName)}'
  location: 'East US 2'       // ✅ CRITICAL: Only this region has sufficient quota
  kind: 'OpenAI'              // ✅ CRITICAL: 'AIServices' requires special approval
  sku: {
    name: 'S0'                // ✅ WORKS: Standard tier that Azure for Students supports
  }
  properties: {
    customSubDomainName: 'ais-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id, resourcePrefix, environmentName)}'
    publicNetworkAccess: 'Enabled'
  }
}

// 2. GPT-4.1-MINI MODEL DEPLOYMENT  
resource gpt41MiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = {
  parent: aiServicesAccount
  name: 'gpt-4.1-mini'        // ✅ EXACT model name required
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1-mini'    // ✅ CRITICAL: Must match deployment name
      version: '2025-04-14'   // ✅ CRITICAL: Only this version works (verified via Azure REST API)
    }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: {
    name: 'GlobalStandard'    // ✅ CRITICAL: 'Standard' SKU fails for this model
    capacity: 1               // ✅ Minimal capacity for cost optimization
  }
}

// 3. EMBEDDINGS MODEL DEPLOYMENT
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = {
  parent: aiServicesAccount
  name: 'text-embedding-ada-002'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'            // ✅ Stable embeddings version
    }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: {
    name: 'Standard'          // ✅ Standard works for embeddings (but not gpt-4.1-mini)
    capacity: 1
  }
  dependsOn: [
    gpt41MiniDeployment       // ✅ Sequential deployment prevents conflicts
  ]
}
```

### **📊 WHY EACH SETTING WORKS (Technical Deep Dive)**

#### **1. `kind: 'OpenAI'` vs `kind: 'AIServices'`**
```bash
# FAILS with Azure for Students:
kind: 'AIServices' 
# Error: "SpecialFeatureOrQuotaIdRequired: The subscription does not have QuotaId/Feature required by SKU 'S0' from kind 'AIServices'"

# WORKS with Azure for Students:
kind: 'OpenAI'
# Reason: OpenAI kind uses different quota allocation that doesn't require special approval
```

**Technical Explanation:**
- `AIServices` = Unified AI services requiring enterprise-grade quota approval
- `OpenAI` = Specific OpenAI service with more permissive quota requirements
- Azure for Students can access OpenAI directly but not unified AI services

#### **2. `version: '2025-04-14'` vs `version: '2024-11-20'`**
```bash
# FAILS - Model version not supported:
version: '2024-11-20'
# Error: "DeploymentModelNotSupported: The model 'Format:OpenAI,Name:gpt-4.1-mini,Version:2024-11-20' is not supported"

# WORKS - Verified via Azure REST API:
version: '2025-04-14'  
# Verification command used: az rest --method GET --uri "https://management.azure.com/subscriptions/.../models?api-version=2023-05-01"
```

**Technical Explanation:**
- Model versions are region-specific and subscription-type-specific
- `2024-11-20` was likely a preview version not available in Azure for Students
- `2025-04-14` is the stable release version available in East US 2 for student subscriptions

#### **3. `sku: 'GlobalStandard'` vs `sku: 'Standard'`**
```bash
# FAILS for gpt-4.1-mini:
sku: { name: 'Standard' }
# Error: "InvalidResourceProperties: The specified SKU 'Standard' of account deployment is not supported by the model 'gpt-4.1-mini'"

# WORKS for gpt-4.1-mini:
sku: { name: 'GlobalStandard' }
# Reason: gpt-4.1-mini requires global deployment SKU, not regional standard
```

**Technical Explanation:**
- Different models require different SKU types
- `gpt-4.1-mini` is a global model requiring `GlobalStandard` SKU
- `text-embedding-ada-002` works with regular `Standard` SKU
- This is model-specific, not subscription-specific

#### **4. `location: 'East US 2'` (Forced Override)**
```bash
# WHY force East US 2:
location: 'East US 2'   // Override user's West US 2 choice
# Reason: Better OpenAI model availability and quota allocation for student subscriptions
```

**Technical Explanation:**
- East US 2 has higher quota allocation for OpenAI services
- Many Azure for Students accounts have better success rates in East US 2
- West US 2 (user's choice) has more restrictive quotas for AI services

### **🚫 CONFIGURATION THAT WILL BREAK DEPLOYMENT**

```bicep
// ❌ THESE WILL CAUSE DEPLOYMENT FAILURES:

// 1. Using AIServices kind (needs enterprise approval)
kind: 'AIServices'          // → SpecialFeatureOrQuotaIdRequired

// 2. Using wrong model version  
version: '2024-11-20'       // → DeploymentModelNotSupported
version: '2024-07-18'       // → DeploymentModelNotSupported (wrong for gpt-4.1-mini)

// 3. Using wrong SKU for gpt-4.1-mini
sku: { name: 'Standard' }   // → InvalidResourceProperties

// 4. Using regions with poor quota
location: 'West Europe'     // → May fail with quota exceeded
location: 'Central US'      // → May fail with quota exceeded
```

### **🔍 HOW WE DISCOVERED THE WORKING CONFIGURATION**

**Step 1: Initial Error Analysis**
```bash
# Original error revealed the core issue:
ERROR: SpecialFeatureOrQuotaIdRequired: The subscription does not have QuotaId/Feature required by SKU 'S0' from kind 'AIServices'
# Solution: Changed kind from 'AIServices' to 'OpenAI'
```

**Step 2: Model Version Discovery**
```bash
# Used Azure REST API to find available versions:
az rest --method GET --uri "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/providers/Microsoft.CognitiveServices/locations/eastus2/models?api-version=2023-05-01" --query "value[?contains(model.name, 'gpt-4.1-mini')].{Name:model.name, Version:model.version}"

# Result showed: gpt-4.1-mini version 2025-04-14 available
```

**Step 3: SKU Testing**
```bash
# Tested different SKUs until GlobalStandard worked:
Standard → FAILED (InvalidResourceProperties)  
GlobalStandard → SUCCESS
```

**Step 4: Regional Optimization**  
```bash
# Tested deployment success rates:
West US 2 → Limited quota
East US 2 → Better quota allocation → SUCCESS
```

### **✅ DEPLOYMENT SUCCESS VERIFICATION**

**These exact settings result in:**
```bash
✅ Done: Azure OpenAI: oai-maintie-rag-prod-yll2wm4u3vm24
✅ Done: Azure AI Services Model Deployment: oai-maintie-rag-prod-yll2wm4u3vm24/gpt-4.1-mini  
✅ Done: Azure AI Services Model Deployment: oai-maintie-rag-prod-yll2wm4u3vm24/text-embedding-ada-002
✅ Done: Container App: ca-backend-maintie-rag-prod
✅ Done: Container App: ca-frontend-maintie-rag-prod
```

### **🔒 PROTECTION AGAINST ACCIDENTAL CHANGES**

**If you need to modify the infrastructure:**
1. **NEVER change** the `kind`, `version`, `sku`, or `location` values above
2. **Test changes** in a separate resource group first
3. **Use `azd provision --preview`** to validate before deploying
4. **Refer to this documentation** before making any AI services changes

**Safe modifications:**
- Resource names (they use uniqueString patterns)
- Tags and metadata
- Capacity values (but keep at 1 for cost optimization)
- Non-AI service configurations

## 🚨 Common Issues Fixed

- **"SpecialFeatureOrQuotaIdRequired"** - Changed from AIServices to OpenAI kind for Azure for Students compatibility
- **"DeploymentModelNotSupported"** - Fixed gpt-4.1-mini version from `2024-11-20` to `2025-04-14`
- **"InvalidResourceProperties" (SKU not supported)** - Changed from `Standard` to `GlobalStandard` SKU for gpt-4.1-mini
- **"VaultAlreadyExists"** - Fixed with stable naming patterns
- **"CustomDomainInUse"** - Fixed with consistent uniqueString  
- **"ServiceQuotaExceeded" (Search)** - Uses Basic tier to avoid free quota conflicts
- **"ServiceQuotaExceeded" (Cosmos DB)** - Disabled free tier (already used in subscription)
- **"InvalidResourceProperties" (OpenAI)** - Updated to 2024-04-01-preview API with sku syntax
- **Resource conflicts** - Removed conflicting modules  
- **Hardcoded references** - All made dynamic

## 🔧 Quick Commands

```bash
# Deploy infrastructure (FULL SUCCESS with working config)
azd up

# Validate Bicep (should show only minor warnings)
az bicep build --file main.bicep

# Test deployment without actually deploying
azd provision --preview

# Check available models in your region (for troubleshooting)
az rest --method GET --uri "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/providers/Microsoft.CognitiveServices/locations/eastus2/models?api-version=2023-05-01" --query "value[?contains(model.name, 'gpt')].{Name:model.name, Version:model.version}"

# Verify deployed models work
az cognitiveservices account deployment list --resource-group rg-maintie-rag-prod --name oai-maintie-rag-prod-yll2wm4u3vm24
```

## 🩺 TROUBLESHOOTING GUIDE

### **If Deployment Fails, Check These in Order:**

#### **1. Quota/Permission Errors**
```bash
# Error: SpecialFeatureOrQuotaIdRequired
# Fix: Verify kind is 'OpenAI' not 'AIServices' in ai-services.bicep line 19

# Error: ServiceQuotaExceeded  
# Fix: Check your Azure for Students quota limits
az account show --query "name" # Should show "Azure for Students"
```

#### **2. Model Version Errors** 
```bash
# Error: DeploymentModelNotSupported
# Fix: Verify model version is exactly '2025-04-14' in ai-services.bicep line 41

# Check what versions are available in your region:
az rest --method GET --uri "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/providers/Microsoft.CognitiveServices/locations/eastus2/models?api-version=2023-05-01" --query "value[?model.name=='gpt-4.1-mini'].{Name:model.name, Version:model.version}"
```

#### **3. SKU Configuration Errors**
```bash
# Error: InvalidResourceProperties (SKU not supported)
# Fix: Verify SKU is 'GlobalStandard' for gpt-4.1-mini deployment in ai-services.bicep line 46

# Error: The specified SKU 'Standard' is not supported
# Fix: gpt-4.1-mini requires 'GlobalStandard', embeddings use 'Standard'
```

#### **4. Regional Issues**
```bash
# Error: Various quota or availability errors
# Fix: Verify location is hardcoded to 'East US 2' in ai-services.bicep line 12

# Check if models are available in different regions:
az rest --method GET --uri "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/providers/Microsoft.CognitiveServices/locations/westus2/models?api-version=2023-05-01" --query "value[?model.name=='gpt-4.1-mini'].{Name:model.name, Version:model.version}"
```

### **Emergency Recovery Commands**

```bash
# If deployment is stuck or failed partially:
az deployment sub cancel --name "prod-$(date +%s)"

# Clean up failed resources (CAREFUL - this deletes everything):
az group delete --name rg-maintie-rag-prod --yes --no-wait

# Start fresh deployment:
azd up
```

### **Validation Commands (Run Before Making Changes)**

```bash
# 1. Validate your subscription type
az account show --query "{Name:name, Type:subscriptionType}" -o table

# 2. Check available models in target region  
az rest --method GET --uri "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/providers/Microsoft.CognitiveServices/locations/eastus2/models?api-version=2023-05-01" --query "value[?contains(model.name, 'gpt-4.1-mini')].{Name:model.name, Version:model.version}" -o table

# 3. Test Bicep compilation
az bicep build --file infra/main.bicep

# 4. Validate template without deploying
az deployment sub validate --location "West US 2" --template-file infra/main.bicep --parameters environmentName=test location="West US 2" principalId="$(az ad signed-in-user show --query id -o tsv)" backendImageName="test" frontendImageName="test"
```

## ⚠️ Expected Warnings

**Container Registry Password Warning:**
```
Warning use-secure-value-for-secure-inputs: Property 'value' expects a secure value
```
This warning is **expected** for Container Apps authentication with Container Registry. The password is properly secured within the Container Apps secrets system.

## 📝 Latest Changes (2025-08-18)

**CRITICAL Azure for Students Quota Fix - FULLY WORKING:**
- **FIXED AI Services quota error** - Changed from `kind: 'AIServices'` to `kind: 'OpenAI'`
- **FIXED SKU requirement** - OpenAI kind works with Azure for Students S0 SKU
- **FIXED location optimization** - Force East US 2 for better OpenAI availability
- **FIXED model version** - Updated gpt-4.1-mini to version `2025-04-14` (not `2024-11-20`)
- **FIXED deployment SKU** - Changed from `Standard` to `GlobalStandard` for gpt-4.1-mini
- **MAINTAINED gpt-4.1-mini model** - Kept requested model deployment
- **✅ DEPLOYMENT SUCCESS** - Full infrastructure deployed including Container Apps
- **✅ AUTOMATED PIPELINE** - postprovision hook executing real data pipeline

**Previous Fixes (2025-01-18):**
- Fixed all naming conflicts with stable uniqueString patterns
- Removed conflicting modules to prevent resource overlaps
- **Fixed Azure Search quota conflict** - Changed from free to basic tier
- **Fixed Cosmos DB quota conflict** - Disabled free tier (already used in subscription)
- **Fixed OpenAI API deprecation** - Updated to 2024-04-01-preview with sku syntax
- **Fixed hardcoded resource references** - ML Workspace now uses proper outputs
- **Fixed AI Services outputs** - Deployment names now dynamic (not hardcoded)
- **Fixed security issue** - Added @secure() to appInsightsConnectionString
- Made all resource references use proper module dependencies
- Updated cost optimization strategy for quota limitations

---

**⚠️ This infrastructure supports a production system. Test thoroughly before changes.**