// Azure Universal RAG Infrastructure - Core Version
// Core resources without Cosmos DB to avoid region availability issues

targetScope = 'resourceGroup'

// Parameters for environment-specific configuration
param environment string = 'dev'
param location string = resourceGroup().location
param resourcePrefix string = 'maintie'

// Add cost optimization parameters - data-driven
param searchSkuName string = (environment == 'prod') ? 'standard' : 'basic'
param storageSkuName string = (environment == 'prod') ? 'Standard_GRS' : 'Standard_LRS'

// Azure Storage Account for Universal RAG data
resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: '${resourcePrefix}${environment}storage'
  location: location
  sku: { name: storageSkuName }  // Environment-driven
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true  // Hierarchical namespace for data lake
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    accessTier: (environment == 'prod') ? 'Hot' : 'Cool'  // Cost optimization
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
  name: '${resourcePrefix}-${environment}-search'
  location: location
  sku: { name: searchSkuName }  // Environment-driven
  properties: {
    replicaCount: (environment == 'prod') ? 2 : 1
    partitionCount: (environment == 'prod') ? 2 : 1
    hostingMode: 'default'
  }
}

// Azure Key Vault for secrets management
resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: '${resourcePrefix}-${environment}-kv'
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Outputs for deployment
output storageAccountName string = storageAccount.name
output searchServiceName string = searchService.name
output keyVaultName string = keyVault.name