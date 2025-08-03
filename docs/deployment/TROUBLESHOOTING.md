# Azure Deployment Troubleshooting Guide

**Common issues and solutions for Azure Universal RAG deployment**

## 🚨 **Common Deployment Issues**

### **1. Cosmos DB Region Capacity Issues**

**Error**: `ServiceUnavailable: Sorry, we are currently experiencing high demand in [region]`

**Solution**:
```bash
# Try alternative regions with better capacity
rm -rf .azure
export AZURE_LOCATION=westus2 && azd up --environment development    # Primary recommendation
export AZURE_LOCATION=westus3 && azd up --environment development    # Alternative 1
export AZURE_LOCATION=centralus && azd up --environment development  # Alternative 2
export AZURE_LOCATION=eastus2 && azd up --environment development    # Alternative 3
```

**Why this happens**: Azure Cosmos DB has regional capacity limits, especially in popular regions like East US.

### **2. Soft-Deleted Resource Conflicts**

**Error**: `FlagMustBeSetForRestore: An existing resource has been soft-deleted. To restore the resource, you must specify 'restore' to be 'true'`

**Solution Option 1 - Purge Soft-Deleted Resources**:
```bash
# Purge soft-deleted OpenAI service
az cognitiveservices account purge --name [service-name] --resource-group [rg-name] --location westus2

# Delete conflicting service principal
az ad sp delete --id [service-principal-id]
```

**Solution Option 2 - Use Different Environment (Recommended)**:
```bash
# Clean deployment state and use fresh environment
rm -rf .azure
export AZURE_LOCATION=westus2
azd up --environment prod
```

**Why this happens**: Azure keeps deleted resources in a soft-delete state for recovery purposes. Using a different environment name creates fresh resources.

### **3. azd Version Issues**

**Error**: Template validation errors or unexpected behavior

**Current Issue**: Using azd 1.12.0 (latest is 1.18.0) without sudo access

**Workaround**:
```bash
# Use environment variables instead of --location flag
export AZURE_LOCATION=westus2
azd up --environment development

# Note: --location flag not available in azd 1.12.0
```

**If you have sudo access**:
```bash
# Upgrade to latest azd version
curl -fsSL https://aka.ms/install-azd.sh | bash
source ~/.bashrc
azd --version  # Should show 1.18.0 or later
```

### **4. Azure ML Workspace Location Mismatch**

**Error**: `AlreadyExistServicePrincipalInDifferentRegion: Location mismatch in AAD and in Model. LocationInAAD: 'eastus', LocationInModel: 'westus2'`

**Solution Option 1 - Delete Service Principal**:
```bash
# Delete conflicting service principal (replace with actual ID from error)
az ad sp delete --id [service-principal-id]
```

**Solution Option 2 - Use Different Environment (Recommended)**:
```bash
rm -rf .azure
export AZURE_LOCATION=westus2
azd up --environment prod
```

**Why this happens**: Service principals created in previous deployments in different regions conflict with new deployments.

### **5. OpenAI Model Deployment Failures**

**Error**: `Azure AI Services Model Deployment failed`

**Cause**: Region doesn't support GPT-4 or quota exceeded

**Solution**:
```bash
# Try regions with better OpenAI availability
export AZURE_LOCATION=westus2 && azd up --environment development
export AZURE_LOCATION=eastus && azd up --environment development
export AZURE_LOCATION=northcentralus && azd up --environment development
```

## 🔧 **Step-by-Step Recovery**

### **If Deployment Partially Fails**

1. **Don't panic** - Partial deployments are common and recoverable

2. **Check what succeeded**:
```bash
az resource list --resource-group rg-maintie-rag-[environment] --output table
```

3. **Option 1 - Retry just the infrastructure**:
```bash
azd provision --force-refresh
```

4. **Option 2 - Clean and retry with same environment**:
```bash
rm -rf .azure
export AZURE_LOCATION=westus2
azd up --environment development
```

5. **Option 3 - Use fresh environment (Recommended for conflicts)**:
```bash
rm -rf .azure
export AZURE_LOCATION=westus2
azd up --environment prod
```

### **If Multiple Regions Fail**

1. **Check Azure Service Health**:
   - Visit: https://status.azure.com/
   - Look for Cosmos DB or OpenAI outages

2. **Try different service tiers**:
   - Cosmos DB: Switch to provisioned instead of serverless
   - OpenAI: Try GPT-3.5 instead of GPT-4

3. **Contact Azure Support** if widespread issues

## 📊 **Regional Recommendations**

### **Best Regions for Azure Universal RAG**

1. **West US 2** ⭐ - Best overall availability
2. **West US 3** - Good alternative  
3. **Central US** - Reliable for most services
4. **East US 2** - Good but sometimes capacity constrained
5. **North Central US** - Good OpenAI availability

### **Regions to Avoid**

- **East US** - Frequently capacity constrained
- **South Central US** - Limited OpenAI models
- **West Europe** - Can have latency issues for US users

## 🛠️ **Advanced Troubleshooting**

### **Check Resource Quotas**

```bash
# Check Cosmos DB quota
az cosmosdb check-name-availability --name your-cosmos-name --type Microsoft.DocumentDB/databaseAccounts

# Check OpenAI quota
az cognitiveservices account list-usage --name your-openai-name --resource-group your-rg
```

### **Manual Resource Creation**

If azd fails repeatedly, create resources manually:

```bash
# Create resource group
az group create --name rg-maintie-rag-development --location westus2

# Create storage account
az storage account create --name stmaintierag$(date +%s) --resource-group rg-maintie-rag-development --location westus2

# Create OpenAI service
az cognitiveservices account create --name oai-maintie-rag --resource-group rg-maintie-rag-development --kind OpenAI --sku S0 --location westus2
```

### **Check Service Dependencies**

Some services have implicit dependencies:

1. **Azure ML** requires **Storage Account** and **Key Vault**
2. **OpenAI** model deployments require **OpenAI service** to be ready
3. **Cosmos DB** Gremlin requires **EnableGremlin** capability

## 📞 **Getting Help**

### **Azure Support**

- **Portal**: portal.azure.com → Help + Support
- **CLI**: `az support -h`
- **Forums**: Microsoft Q&A for Azure

### **Project Support**

- **Documentation**: `docs/getting-started/QUICK_START.md`
- **Implementation Plan**: `docs/development/LOCAL_TESTING_IMPLEMENTATION_PLAN.md`
- **Architecture**: `docs/architecture/SYSTEM_ARCHITECTURE.md`

## ✅ **Deployment Success Indicators**

When deployment succeeds, you should see:

```
✅ Resource group: rg-maintie-rag-[environment]
✅ Storage account: stmaintierag[unique]
✅ Azure OpenAI: oai-maintie-rag-[environment]-[unique]
✅ Azure Search: srch-maintie-rag-[environment]-[unique]
✅ Azure Cosmos DB: cosmos-maintie-rag-[environment]-[unique]
✅ Azure ML Workspace: ml-maintierag-[unique]
✅ Key Vault: kv-maintierag[environment]-[unique]
✅ Container Apps: Backend deployed successfully
```

**Supported Environment Names**:
- `prod` / `production` - Premium tier (2+ replicas, premium storage)
- `staging` - Standard tier (1-5 replicas, standard storage)  
- `development` - Basic tier (1-3 replicas, basic storage)
- `test` - Basic tier (same as development)

## 🔧 **Manual Deployment Steps**

After infrastructure deployment (80% automated), complete these manual steps:

### **1. Deploy OpenAI Models Manually**

**Required due to API compatibility issues with newer models**

1. **Navigate to Azure OpenAI Service**:
   - Go to Azure Portal → Resource Groups → `rg-maintie-rag-[environment]`
   - Click on `oai-maintie-rag-[environment]-[unique]`
   - Go to "Model deployments" section

2. **Deploy GPT-4.1 Model**:
   ```
   Model: gpt-4.1
   Deployment name: gpt-4
   Model version: 2025-04-14
   Deployment type: Global Standard
   Capacity: 250K tokens per minute (TPM)
   ```

3. **Deploy GPT-4.1 Mini Model** (for production environments):
   ```
   Model: gpt-4.1-mini
   Deployment name: gpt-4-mini
   Model version: 2025-04-14
   Deployment type: Global Standard
   Capacity: 250K tokens per minute (TPM)
   ```

4. **Deploy Text Embedding Model**:
   ```
   Model: text-embedding-ada-002
   Deployment name: text-embedding-ada-002
   Model version: 2
   Deployment type: Global Standard
   Capacity: 250K tokens per minute (TPM)
   ```

### **2. Deploy Azure ML Workspace (Optional)**

**Required only for GNN training functionality**

1. **Navigate to Resource Group**:
   - Go to Azure Portal → Resource Groups → `rg-maintie-rag-[environment]`
   - Click "Create" → Search "Machine Learning"

2. **Create ML Workspace**:
   ```
   Name: ml-maintierag-[unique]
   Subscription: Microsoft Azure Sponsorship
   Resource group: rg-maintie-rag-[environment]
   Location: West US 2
   Storage account: stmaintier[unique] (select existing)
   Key vault: kv-maintieragp-[unique] (select existing)
   Application insights: appi-maintie-rag-[environment] (select existing)
   ```

3. **Configure Compute Resources**:
   - Create compute cluster for GNN training
   - Instance type: Standard_DS3_v2 (development) or Standard_DS4_v2 (production)
   - Min nodes: 0, Max nodes: 1-10 (based on environment)

## 🚀 **After Manual Deployment Complete**

Test the deployed system:

```bash
# Test Azure services connectivity
python scripts/test_azure_connectivity.py

# Test complete data pipeline
python scripts/test_data_pipeline.py

# Test tri-modal search
python scripts/test_tri_modal_search.py
```

---

**Remember**: Azure deployments can be temperamental. Don't hesitate to retry with different regions if you encounter capacity issues!