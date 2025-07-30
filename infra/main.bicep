// Azure Universal RAG Infrastructure - Main Entry Point
// Azure Developer CLI (azd) compatible infrastructure
targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment (e.g., dev, staging, prod)')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Principal ID for the current user (from azd auth)')
param principalId string = ''

// Resource naming configuration
var resourcePrefix = 'maintie-rag'
var resourceGroupName = 'rg-${resourcePrefix}-${environmentName}'

// Create resource group
resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: location
  tags: {
    Environment: environmentName
    Project: 'Azure Universal RAG'
    Purpose: 'Maintenance Knowledge Graph + Vector Search + GNN Training'
    DeployedBy: 'azd'
    CreatedDate: '2025-07-28'
  }
}

// Deploy core infrastructure modules
module coreServices 'modules/core-services.bicep' = {
  name: 'coreServices'
  scope: resourceGroup
  params: {
    environmentName: environmentName
    location: location
    principalId: principalId
    resourcePrefix: resourcePrefix
  }
}

module ai 'modules/ai-services.bicep' = {
  name: 'aiServices'
  scope: resourceGroup
  params: {
    environmentName: environmentName
    location: location
    principalId: principalId
    resourcePrefix: resourcePrefix
    managedIdentityPrincipalId: coreServices.outputs.managedIdentityPrincipalId
  }
}

module data 'modules/data-services.bicep' = {
  name: 'dataServices'
  scope: resourceGroup
  params: {
    environmentName: environmentName
    location: location
    principalId: principalId
    resourcePrefix: resourcePrefix
    managedIdentityPrincipalId: coreServices.outputs.managedIdentityPrincipalId
  }
}

module hosting 'modules/hosting-services.bicep' = {
  name: 'hostingServices'
  scope: resourceGroup
  params: {
    environmentName: environmentName
    location: location
    principalId: principalId
    resourcePrefix: resourcePrefix
    openaiEndpoint: ai.outputs.openaiEndpoint
    searchEndpoint: coreServices.outputs.searchEndpoint
    cosmosEndpoint: data.outputs.cosmosEndpoint
    storageAccountName: coreServices.outputs.storageAccountName
    keyVaultName: coreServices.outputs.keyVaultName
    appInsightsConnectionString: coreServices.outputs.appInsightsConnectionString
  }
}

// Outputs for azd and application configuration
output AZURE_LOCATION string = location
output AZURE_RESOURCE_GROUP string = resourceGroup.name

// AI Services
output AZURE_OPENAI_ENDPOINT string = ai.outputs.openaiEndpoint
output AZURE_OPENAI_DEPLOYMENT_NAME string = ai.outputs.deploymentName
output AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME string = ai.outputs.embeddingDeploymentName

// Data Services
output AZURE_SEARCH_ENDPOINT string = coreServices.outputs.searchEndpoint
output AZURE_SEARCH_INDEX string = 'maintie-${environmentName}-index'
output AZURE_COSMOS_ENDPOINT string = data.outputs.cosmosEndpoint
output AZURE_COSMOS_DATABASE_NAME string = 'maintie-rag-${environmentName}'
output AZURE_COSMOS_GRAPH_NAME string = 'knowledge-graph-${environmentName}'

// Storage & Security
output AZURE_STORAGE_ACCOUNT string = coreServices.outputs.storageAccountName
output AZURE_STORAGE_CONTAINER string = 'maintie-${environmentName}-data'
output AZURE_KEY_VAULT_NAME string = coreServices.outputs.keyVaultName

// ML Services
output AZURE_ML_WORKSPACE_NAME string = data.outputs.mlWorkspaceName
output AZURE_ML_RESOURCE_GROUP string = resourceGroup.name

// Monitoring
output AZURE_APP_INSIGHTS_CONNECTION_STRING string = coreServices.outputs.appInsightsConnectionString
output AZURE_LOG_ANALYTICS_WORKSPACE_ID string = coreServices.outputs.logAnalyticsWorkspaceId

// Hosting
output SERVICE_BACKEND_URI string = hosting.outputs.backendUri
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = hosting.outputs.registryLoginServer
output AZURE_CONTAINER_ENVIRONMENT_NAME string = hosting.outputs.containerEnvironmentName

// Identity
output AZURE_CLIENT_ID string = coreServices.outputs.managedIdentityClientId
output AZURE_TENANT_ID string = tenant().tenantId
output AZURE_SUBSCRIPTION_ID string = subscription().subscriptionId