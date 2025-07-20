// Azure Universal RAG Infrastructure - Working Services Only
// Only includes services that are successfully deployed and operational

targetScope = 'resourceGroup'

// Parameters for environment-specific configuration
param environment string = 'dev'
param location string = resourceGroup().location
param resourcePrefix string = 'maintie'

// Data-driven resource configuration by environment
var resourceConfig = {
  dev: {
    searchSku: 'basic'
    searchReplicas: 1
    searchPartitions: 1
    storageSku: 'Standard_LRS'
    storageAccessTier: 'Cool'
    keyVaultSku: 'standard'
    appInsightsSampling: 10
    retentionDays: 30
  }
  staging: {
    searchSku: 'standard'
    searchReplicas: 1
    searchPartitions: 1
    storageSku: 'Standard_ZRS'
    storageAccessTier: 'Hot'
    keyVaultSku: 'standard'
    appInsightsSampling: 5
    retentionDays: 60
  }
  prod: {
    searchSku: 'standard'
    searchReplicas: 2
    searchPartitions: 2
    storageSku: 'Standard_GRS'
    storageAccessTier: 'Hot'
    keyVaultSku: 'premium'
    appInsightsSampling: 1
    retentionDays: 90
  }
}

// Get current environment configuration
var currentConfig = resourceConfig[environment]

// Simple deterministic naming with deployment token
param deploymentToken string = uniqueString(resourceGroup().id, deployment().name)

// Deterministic resource naming
var storageAccountName = '${resourcePrefix}${environment}stor${take(deploymentToken, 8)}'
var searchServiceName = '${resourcePrefix}-${environment}-search-${take(deploymentToken, 6)}'
var keyVaultName = '${resourcePrefix}-${environment}-kv-${take(deploymentToken, 6)}'

// Azure Storage Account for Universal RAG data
resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: storageAccountName
  location: location
  sku: { name: currentConfig.storageSku }
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true  // Hierarchical namespace for data lake
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    accessTier: currentConfig.storageAccessTier
  }
}

// Blob service for Universal RAG data
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2021-04-01' = {
  parent: storageAccount
  name: 'default'
}

// Storage container for Universal RAG data
resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2021-04-01' = {
  parent: blobService
  name: 'universal-rag-data'
  properties: {
    publicAccess: 'None'
  }
}

// Azure Cognitive Search for vector indices
resource searchService 'Microsoft.Search/searchServices@2020-08-01' = {
  name: searchServiceName
  location: location
  sku: { name: currentConfig.searchSku }
  properties: {
    replicaCount: currentConfig.searchReplicas
    partitionCount: currentConfig.searchPartitions
    hostingMode: 'default'
  }
}

// Azure Key Vault for secrets management
resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: currentConfig.keyVaultSku
    }
    tenantId: subscription().tenantId
    accessPolicies: []
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Azure Application Insights for monitoring
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${resourcePrefix}-${environment}-appinsights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
    SamplingPercentage: currentConfig.appInsightsSampling
  }
}

// Log Analytics Workspace for monitoring
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${resourcePrefix}-${environment}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: currentConfig.retentionDays
  }
}

// Outputs for deployment
output storageAccountName string = storageAccount.name
output storageAccountId string = storageAccount.id
output searchServiceName string = searchService.name
output searchServiceId string = searchService.id
output keyVaultName string = keyVault.name
output keyVaultId string = keyVault.id
output deploymentToken string = deploymentToken
output applicationInsightsName string = applicationInsights.name
output logAnalyticsWorkspaceName string = logAnalyticsWorkspace.name
output environmentConfig object = currentConfig