// Azure Universal RAG ML Infrastructure - Simplified (ML Storage + ML Workspace only)
// Deploys: ML Storage Account, ML Workspace
// Skips: Cosmos DB, Container Environment, Container App (due to region availability)

targetScope = 'resourceGroup'

// Parameters for environment-specific configuration
param environment string = 'dev'
param location string = resourceGroup().location
param resourcePrefix string = 'maintie'

// Data-driven resource configuration by environment - from existing codebase pattern
var resourceConfig = {
  dev: {
    mlComputeInstances: 1
    mlVmSize: 'Standard_DS2_v2'
    storageSku: 'Standard_LRS'
    storageAccessTier: 'Cool'
  }
  staging: {
    mlComputeInstances: 2
    mlVmSize: 'Standard_DS3_v2'
    storageSku: 'Standard_ZRS'
    storageAccessTier: 'Hot'
  }
  prod: {
    mlComputeInstances: 4
    mlVmSize: 'Standard_DS4_v2'
    storageSku: 'Standard_GRS'
    storageAccessTier: 'Hot'
  }
}

// Get current environment configuration
var currentConfig = resourceConfig[environment]

// Use same deterministic naming pattern as core template
@minLength(10)
@description('Unique deployment token from main deployment')
param deploymentToken string = uniqueString(resourceGroup().id, environment, resourcePrefix)

// Deterministic resource naming - following existing pattern
var mlStorageAccountName = '${resourcePrefix}${environment}mlstor${take(deploymentToken, 8)}'
var mlWorkspaceName = '${resourcePrefix}-${environment}-ml'

// Parameters for existing resources from core deployment
param existingStorageAccountName string
param existingKeyVaultName string
param existingAppInsightsName string = '${resourcePrefix}-${environment}-appinsights'

// Get existing resources from core deployment
resource existingStorageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' existing = {
  name: existingStorageAccountName
}

resource existingKeyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' existing = {
  name: existingKeyVaultName
}

resource existingAppInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: existingAppInsightsName
}

// ML Storage Account (for ML workspace data)
resource mlStorageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: mlStorageAccountName
  location: location
  sku: { name: currentConfig.storageSku }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    accessTier: currentConfig.storageAccessTier
  }
}

// ML Storage Blob Service
resource mlBlobService 'Microsoft.Storage/storageAccounts/blobServices@2021-04-01' = {
  parent: mlStorageAccount
  name: 'default'
}

// ML Workspace Container
resource mlWorkspaceContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2021-04-01' = {
  parent: mlBlobService
  name: 'azureml'
  properties: {
    publicAccess: 'None'
  }
}

// Azure ML Workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2021-04-01' = {
  name: mlWorkspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    storageAccount: mlStorageAccount.id
    keyVault: existingKeyVault.id
    applicationInsights: existingAppInsights.id
    friendlyName: 'Universal RAG ML Workspace - ${toUpper(environment)}'
    description: 'ML workspace for Universal RAG GNN training and inference'
  }
}

// Role assignments for ML workspace managed identity
resource storageContributorRoleDefinition 'Microsoft.Authorization/roleDefinitions@2018-01-01-preview' existing = {
  scope: subscription()
  name: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
}

resource mlWorkspaceStorageRoleAssignment 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: mlStorageAccount
  name: guid(mlWorkspace.id, mlStorageAccount.id, storageContributorRoleDefinition.id)
  properties: {
    roleDefinitionId: storageContributorRoleDefinition.id
    principalId: mlWorkspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Outputs for verification - following existing pattern
output mlStorageAccountName string = mlStorageAccount.name
output mlStorageAccountId string = mlStorageAccount.id
output mlWorkspaceName string = mlWorkspace.name
output mlWorkspaceId string = mlWorkspace.id
output deploymentToken string = deploymentToken
output environmentConfig object = currentConfig