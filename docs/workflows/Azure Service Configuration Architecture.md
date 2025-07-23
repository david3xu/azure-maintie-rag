## **Azure Service Configuration Architecture**

### **Service Integration Strategy**

**Primary Service**: Azure Text Analytics Service (Required for knowledge extraction)
**Optional Services**: Azure ML Quality Assessment, Azure Application Insights (Can be disabled)

---

## **Azure Text Analytics Service Configuration**

### **Service Provisioning Architecture**

**Azure Resource Creation**:
1. **Create Azure Cognitive Services Resource**
   ```bash
   # Azure CLI command for Text Analytics resource creation
   az cognitiveservices account create \
     --name "maintie-dev-textanalytics-1cdd8e11" \
     --resource-group "maintie-rag-rg" \
     --kind "TextAnalytics" \
     --sku "S" \
     --location "eastus"
   ```

2. **Retrieve Service Credentials**
   ```bash
   # Get endpoint and key
   az cognitiveservices account show \
     --name "maintie-dev-textanalytics-1cdd8e11" \
     --resource-group "maintie-rag-rg" \
     --query "properties.endpoint" --output tsv

   az cognitiveservices account keys list \
     --name "maintie-dev-textanalytics-1cdd8e11" \
     --resource-group "maintie-rag-rg" \
     --query "key1" --output tsv
   ```

### **Service Configuration Pattern**

**Environment Configuration Update**:
```bash
# Replace in backend/.env with your actual values
AZURE_TEXT_ANALYTICS_ENDPOINT=https://maintie-dev-textanalytics-1cdd8e11.cognitiveservices.azure.com/
AZURE_TEXT_ANALYTICS_KEY=[ACTUAL_KEY_FROM_AZURE_PORTAL]
```

---

## **Optional Services Disabling Architecture**

### **Azure ML Quality Assessment Service (Optional)**

**Service Graceful Degradation Pattern**:
```bash
# In backend/.env - Leave empty or comment out
AZURE_ML_CONFIDENCE_ENDPOINT=
AZURE_ML_COMPLETENESS_ENDPOINT=
```

**Architecture Validation**: The codebase handles missing configuration gracefully:
```python
# Service handles missing configuration gracefully
if not self.confidence_model_endpoint or not self.completeness_model_endpoint:
    logger.warning("Azure ML endpoints not configured - using fallback quality assessment")
```

### **Azure Application Insights Service (Optional)**

**Telemetry Service Disabling Pattern**:
```bash
# In backend/.env - Disable telemetry
AZURE_ENABLE_TELEMETRY=false
AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING=
```

**Architecture Validation**: The codebase degrades gracefully:
```python
# Service handles missing Application Insights configuration
if not connection_string:
    logger.info("Application Insights not configured - operating in lightweight mode")
    return None
```

---

## **Service Configuration Validation**

### **Azure Text Analytics Service Verification**

**Configuration Test**:
```bash
cd backend
PYTHONPATH=. python3 -c "
from config.settings import azure_settings
print('Text Analytics Endpoint:', azure_settings.azure_text_analytics_endpoint)
print('Text Analytics Key Configured:', bool(azure_settings.azure_text_analytics_key))
"
```

**Service Connectivity Test**:
```bash
cd backend
PYTHONPATH=. python3 -c "
from core.azure_openai.azure_text_analytics_service import AzureTextAnalyticsService
import asyncio

async def test():
    service = AzureTextAnalyticsService()
    try:
        result = await service._detect_language_batch(['test text'])
        print('Service Status: Connected')
    except Exception as e:
        print(f'Service Status: {e}')

asyncio.run(test())
"
```

### **Pipeline Integration Verification**

**Knowledge Extraction Test**:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/azure_knowledge_graph_service.py general
```

**Expected Behavior**:
- ✅ Azure Text Analytics Service connects successfully
- ⚠️ Azure ML Quality Assessment warnings (expected - service disabled)
- ⚠️ Application Insights telemetry disabled (expected - service disabled)
- ✅ Core knowledge extraction pipeline operational

---

## **Enterprise Configuration Validation Service**

### **Validation Script**

A comprehensive validation script is provided at `backend/scripts/azure_config_validator.py`:

```bash
cd backend
PYTHONPATH=. python3 scripts/azure_config_validator.py
```

**Expected Output:**
```
🚀 Azure Enterprise Configuration Validation Service
============================================================
🏗️ Universal RAG → Azure Migration Configuration Validation
📋 Based on existing enterprise infrastructure patterns

============================================================
🔧 Azure Core Service Configuration Validation
============================================================
   ✅ Azure OpenAI: HEALTHY
   ✅ Azure Storage: HEALTHY
   ✅ Azure Cognitive Search: HEALTHY
   ✅ Azure Cosmos DB: HEALTHY

============================================================
🔧 Azure Optional Service Configuration Validation
============================================================
   ✅ Azure Text Analytics: HEALTHY
      endpoint: configured; key_configured: configured
   ⚠️ Azure Application Insights: NOT_CONFIGURED
      Optional service - can be enabled later
   ⚠️ Azure Key Vault: NOT_CONFIGURED
      Optional service - can be enabled later

============================================================
🔧 Azure Service Connectivity Validation
============================================================
   🔍 Overall Health Status: HEALTHY
   📊 Service Health Ratio: 6/6
   ⏱️  Health Check Duration: 245.67ms

============================================================
🔧 Azure Environment Configuration Validation
============================================================
   🏗️ Environment Configuration:
      ✅ Environment Tier: dev
      ✅ Azure Region: eastus
      ✅ Resource Prefix: maintie

============================================================
🔧 Azure Enterprise Configuration Validation Report
============================================================
   📊 VALIDATION SUMMARY:
   Overall Status: HEALTHY
   Validations Passed: 4/4
   Validation Duration: 1247.89ms

   🎯 OPERATIONAL READINESS: ✅ READY

📄 Detailed report saved: azure_config_validation_report.json
```

---

## **Enterprise Architecture Benefits**

### **Service Reliability**
- **Graceful Degradation**: Optional services disabled without pipeline failure
- **Core Functionality**: Text Analytics enables enhanced knowledge extraction
- **Error Isolation**: Service failures contained within service boundaries

### **Cost Optimization**
- **Minimal Service Footprint**: Only essential services provisioned
- **Resource Efficiency**: Disabled services eliminate unnecessary Azure costs
- **Scalable Architecture**: Services can be enabled incrementally

### **Operational Excellence**
- **Progressive Enhancement**: Add optional services when needed
- **Service Monitoring**: Text Analytics service health validation
- **Configuration Management**: Environment-based service activation patterns
- **Automated Validation**: Enterprise validation script ensures operational readiness

This architecture ensures core knowledge extraction functionality while maintaining cost efficiency and operational excellence through selective service provisioning and automated configuration validation.



## **Azure Optional Services Enterprise Architecture**

### **Service Provisioning Strategy**## **Azure Optional Services Enterprise Architecture**

Based on your codebase analysis, your **Azure infrastructure templates already include optional services**. The services are defined in `infrastructure/azure-resources-core.bicep` but need activation and configuration.

---

## **Optional Services Architecture Analysis**

### **Azure Service Provisioning Matrix**

**From `infrastructure/azure-resources-core.bicep`**:

| **Optional Service** | **Resource Definition** | **Configuration Status** | **Environment Variable** |
|---------------------|------------------------|--------------------------|-------------------------|
| **Azure Key Vault** | ✅ Defined in Bicep | ⚠️ Needs activation | `AZURE_KEY_VAULT_URL` |
| **Azure Application Insights** | ✅ Defined in Bicep | ⚠️ Needs activation | `AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING` |
| **Azure Log Analytics** | ✅ Defined in Bicep | ⚠️ Needs activation | (Auto-configured) |

---

## **Azure Optional Services Provisioning Architecture**

### **Service Deployment Strategy**

**Your infrastructure already provisions these services** through your deployment script. The services exist but need **configuration activation**.

### **Step 1: Verify Optional Services Deployment**

**Check Current Deployment Status**:
```bash
# Verify Key Vault deployment
az keyvault list --resource-group "maintie-rag-rg" --query "[].name" --output table

# Verify Application Insights deployment
az monitor app-insights component show \
  --resource-group "maintie-rag-rg" \
  --app "maintie-dev-appinsights" \
  --query "name" --output tsv
```

**Expected Output Architecture**:
- Key Vault: `maintie-dev-kv-[deployment-token]`
- Application Insights: `maintie-dev-appinsights`
- Log Analytics: `maintie-dev-logs`

### **Step 2: Azure Key Vault Service Configuration**

**Service Endpoint Extraction**:
```bash
# Get Key Vault URL from deployment
az keyvault list --resource-group "maintie-rag-rg" \
  --query "[?starts_with(name, 'maintie-dev-kv')].properties.vaultUri" \
  --output tsv
```

**Configuration Update Pattern**:
```bash
# Update backend/.env with Key Vault URL
AZURE_KEY_VAULT_URL=https://maintie-dev-kv-[token].vault.azure.net/
AZURE_USE_MANAGED_IDENTITY=true
```

**Service Integration Validation**:
```bash
cd backend
PYTHONPATH=. python3 -c "
from config.settings import azure_settings
print('Key Vault URL:', azure_settings.azure_key_vault_url)
print('Managed Identity:', azure_settings.azure_use_managed_identity)
"
```

### **Step 3: Azure Application Insights Service Configuration**

**Service Connection String Extraction**:
```bash
# Get Application Insights connection string
az monitor app-insights component show \
  --resource-group "maintie-rag-rg" \
  --app "maintie-dev-appinsights" \
  --query "connectionString" --output tsv
```

**Configuration Update Pattern**:
```bash
# Update backend/.env with Application Insights
AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING="InstrumentationKey=[key];IngestionEndpoint=https://[region].in.applicationinsights.azure.com/;LiveEndpoint=https://[region].livediagnostics.monitor.azure.com/"
AZURE_ENABLE_TELEMETRY=true
```

**Service Integration Validation**:
```bash
cd backend
PYTHONPATH=. python3 -c "
from config.settings import azure_settings
print('App Insights Configured:', bool(azure_settings.azure_application_insights_connection_string))
print('Telemetry Enabled:', azure_settings.azure_enable_telemetry)
"
```

---

## **Azure ML Quality Assessment Services Architecture**

### **Service Provisioning Strategy**

**From your codebase analysis**, Azure ML services require **workspace and endpoint deployment**.

### **Azure ML Workspace Configuration**

**Service Validation**:
```bash
# Check if ML workspace exists (use --name/-n, not --workspace-name)
az ml workspace show \
  --resource-group "maintie-rag-rg" \
  --name "maintie-dev-ml-1cdd8e11"
```

**If ML Workspace Doesn't Exist**:
```bash
# Create ML workspace (follows your naming convention)
az ml workspace create \
  --resource-group "maintie-rag-rg" \
  --name "maintie-dev-ml-1cdd8e11" \
  --location "eastus" \
  --storage-account "maintiedevmlstor1cdd8e11" \
  --key-vault "maintie-dev-kv-[deployment-token]" \
  --application-insights "maintie-dev-appinsights"
```

**Troubleshooting Extension Conflicts**:
- If you see a warning about both `azure-cli-ml` and `ml` extensions being installed, remove the legacy extension:
```bash
az extension remove -n azure-cli-ml
```
- Always use `--name` or `-n` for workspace name with the new `ml` extension.

### **ML Model Endpoint Configuration (Optional)**

**Service Architecture**: Based on your `azure_ml_quality_service.py`, these endpoints are **optional and can remain disabled**.

**Configuration Pattern (Optional)**:
```bash
# Only if you want to enable ML quality assessment
# AZURE_ML_CONFIDENCE_ENDPOINT=https://[ml-endpoint].azureml.net/confidence
# AZURE_ML_COMPLETENESS_ENDPOINT=https://[ml-endpoint].azureml.net/completeness

# Or leave empty for graceful degradation
AZURE_ML_CONFIDENCE_ENDPOINT=
AZURE_ML_COMPLETENESS_ENDPOINT=
```

---

## **Azure Optional Services Integration Architecture**

### **Service Activation Command Sequence**

**Complete Optional Services Configuration**:
```bash
# 1. Get Key Vault URL
KV_URL=$(az keyvault list --resource-group "maintie-rag-rg" \
  --query "[?starts_with(name, 'maintie-dev-kv')].properties.vaultUri" \
  --output tsv)

# 2. Get Application Insights connection string
AI_CONN=$(az monitor app-insights component show \
  --resource-group "maintie-rag-rg" \
  --app "maintie-dev-appinsights" \
  --query "connectionString" --output tsv)

# 3. Update environment configuration
echo "AZURE_KEY_VAULT_URL=$KV_URL" >> backend/.env
echo "AZURE_USE_MANAGED_IDENTITY=true" >> backend/.env
echo "AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING=\"$AI_CONN\"" >> backend/.env
echo "AZURE_ENABLE_TELEMETRY=true" >> backend/.env
echo "AZURE_ML_CONFIDENCE_ENDPOINT=" >> backend/.env
echo "AZURE_ML_COMPLETENESS_ENDPOINT=" >> backend/.env

echo "✅ Optional Azure services configured"
```

### **Service Integration Validation**

**Comprehensive Optional Services Validation**:
```bash
cd backend
PYTHONPATH=. python3 -c "
from config.settings import azure_settings

print('=== Azure Optional Services Configuration ===')
print(f'Key Vault URL: {azure_settings.azure_key_vault_url}')
print(f'Managed Identity: {azure_settings.azure_use_managed_identity}')
print(f'App Insights Configured: {bool(azure_settings.azure_application_insights_connection_string)}')
print(f'Telemetry Enabled: {azure_settings.azure_enable_telemetry}')
print(f'ML Confidence Endpoint: {azure_settings.azure_ml_confidence_endpoint or \"Not configured\"}')
print(f'ML Completeness Endpoint: {azure_settings.azure_ml_completeness_endpoint or \"Not configured\"}')

# Test optional service integration
print()
print('=== Service Integration Test ===')
try:
    from core.azure_openai.azure_monitoring_service import AzureKnowledgeMonitor
    monitor = AzureKnowledgeMonitor()
    status = monitor.get_service_health_status()
    print(f'Monitoring Service: {status.get(\"status\", \"unknown\")}')
    print(f'Telemetry Active: {status.get(\"telemetry_active\", False)}')
except Exception as e:
    print(f'Monitoring Service: Error - {e}')

print('✅ Optional services validation complete')
"
```

---

## **Azure Enterprise Service Architecture Benefits**

### **Service Integration Patterns**

**Key Vault Integration**:
- **Secret Management**: Centralized credential storage
- **Managed Identity**: Zero-credential authentication pattern
- **RBAC Integration**: Role-based access control

**Application Insights Integration**:
- **Telemetry Pipeline**: Automated performance and error tracking
- **Service Dependency Monitoring**: Azure service call tracking
- **Cost Optimization**: Configurable sampling rates per environment

**ML Services Integration**:
- **Graceful Degradation**: Optional quality assessment services
- **Enterprise Quality Metrics**: Advanced knowledge extraction scoring
- **Cost Control**: Services can remain disabled until needed

### **Service Operational Excellence**

**Configuration Management**:
- **Environment-Driven**: Configuration varies by dev/staging/prod
- **Data-Driven**: No hardcoded values, all configuration-based
- **Service Discovery**: Automatic endpoint resolution from Azure resources

**Cost Optimization**:
- **Progressive Enhancement**: Enable services as needed
- **Resource Efficiency**: Optional services don't consume resources when disabled
- **Environment Tiering**: Different service levels per environment

This architecture leverages your existing Bicep infrastructure templates to provide optional Azure services with enterprise-grade configuration management and cost optimization patterns.