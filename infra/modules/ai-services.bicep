// AI Services Module: Azure OpenAI with gpt-4.1-mini deployment
param environmentName string
param location string
param principalId string
param resourcePrefix string
param managedIdentityPrincipalId string

// Single configuration - PIPELINE OPTIMIZED (HIGHER CAPACITY for dataflow)
var config = {
  miniCapacity: 30 // INCREASED: 30K TPM for multi-phase dataflow pipeline (was 10K TPM)
  embeddingCapacity: 20 // INCREASED: 20K TPM for vector embeddings (was 10K TPM)
  location: 'East US 2' // Force East US 2 for better OpenAI quota availability in Azure for Students
}

// Azure OpenAI - Compatible with Azure for Students (S0 SKU requires special quota)
resource aiServicesAccount 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name: 'oai-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id, resourcePrefix, environmentName)}'
  location: config.location
  kind: 'OpenAI'
  sku: {
    name: 'S0' // Standard tier - use location with better quota availability
  }
  properties: {
    customSubDomainName: 'ais-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id, resourcePrefix, environmentName)}'
    publicNetworkAccess: 'Enabled'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Azure OpenAI with gpt-4.1-mini'
  }
}

// GPT-4.1-mini Model Deployment (as specifically requested)
resource gpt41MiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = {
  parent: aiServicesAccount
  name: 'gpt-4.1-mini'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1-mini'
      version: '2025-04-14'
    }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: {
    name: 'GlobalStandard'
    capacity: config.miniCapacity // MINIMAL capacity per README
  }
}

// Text Embedding Model
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = {
  parent: aiServicesAccount
  name: 'text-embedding-ada-002'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
    }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: {
    name: 'Standard'
    capacity: config.embeddingCapacity
  }
  dependsOn: [
    gpt41MiniDeployment
  ]
}

// Outputs
output openaiAccountName string = aiServicesAccount.name
output openaiEndpoint string = aiServicesAccount.properties.endpoint
output openaiResourceId string = aiServicesAccount.id

output deploymentName string = gpt41MiniDeployment.name
output embeddingDeploymentName string = embeddingDeployment.name

output openaiLocation string = config.location
output openaiResourceGroup string = resourceGroup().name
