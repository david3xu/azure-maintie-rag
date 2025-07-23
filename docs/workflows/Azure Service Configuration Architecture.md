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