// Core Services Module: Storage, Search, KeyVault, Monitoring, Identity
param environmentName string
param location string
param principalId string
param resourcePrefix string

// Single configuration - FREE TIER OPTIMIZED (Cost Savings)
var config = {
  searchSku: 'basic' // BASIC: Avoid free tier quota conflicts (existing free service)
  searchReplicas: 1
  searchPartitions: 1
  storageSku: 'Standard_LRS' // CHEAPEST: Locally redundant storage
  storageAccessTier: 'Cool' // CHEAPER: Lower storage costs
  keyVaultSku: 'standard' // FREE: 10,000 operations/month
  logRetentionDays: 30 // AZURE FOR STUDENTS: Minimum allowed retention
  // Container Registry: managed by hosting-services module
}

// Managed Identity for all services
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${resourcePrefix}-${environmentName}'
  location: location
  tags: {
    Environment: environmentName
    Purpose: 'Universal RAG System Identity'
  }
}

// Log Analytics Workspace - add timestamp to avoid soft-delete conflicts
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${resourcePrefix}-${environmentName}-${take(uniqueString(resourceGroup().id, resourcePrefix, environmentName, '2025'), 8)}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: config.logRetentionDays // AZURE FOR STUDENTS: Use allowed retention
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
  tags: {
    Environment: environmentName
    Purpose: 'Centralized logging for Universal RAG'
  }
}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${resourcePrefix}-${environmentName}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Application performance monitoring'
  }
}

// Azure Cognitive Search - Use Basic tier to avoid quota conflicts
resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: 'srch-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id, resourcePrefix, environmentName)}'
  location: location
  sku: {
    name: config.searchSku // Use config setting (basic tier to avoid free quota conflicts)
  }
  properties: {
    replicaCount: config.searchReplicas
    partitionCount: config.searchPartitions
    hostingMode: 'default'
    publicNetworkAccess: 'enabled'
    semanticSearch: 'disabled' // DISABLE: Free tier doesn't support semantic search
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Vector search and full-text search for maintenance data'
  }
}

// Storage Account - Multi-container setup for different data types
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'st${take(replace(replace('${resourcePrefix}${environmentName}', '-', ''), '_', ''), 8)}${take(uniqueString(resourceGroup().id, resourcePrefix, environmentName), 10)}'
  location: location
  sku: {
    name: config.storageSku
  }
  kind: 'StorageV2'
  properties: {
    accessTier: config.storageAccessTier
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    defaultToOAuthAuthentication: false
    dnsEndpointType: 'Standard'
    minimumTlsVersion: 'TLS1_2'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
    }
    publicNetworkAccess: 'Enabled'
    supportsHttpsTrafficOnly: true
  }
  identity: {
    type: 'SystemAssigned'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Data storage for RAG system (raw data, processed data, models)'
  }
}

// Storage containers for organized data management
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    cors: {
      corsRules: []
    }
    deleteRetentionPolicy: {
      allowPermanentDelete: false
      enabled: true
      days: 7 // MINIMUM: Always use 7 days to save costs
    }
  }
}

// Data containers
resource rawDataContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'maintie-${environmentName}-rawdata'
  properties: {
    publicAccess: 'None'
  }
}

resource processedDataContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'maintie-${environmentName}-processed'
  properties: {
    publicAccess: 'None'
  }
}

resource modelsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'maintie-${environmentName}-models'
  properties: {
    publicAccess: 'None'
  }
}

// Key Vault for secrets management - Use timestamp to avoid soft-delete conflicts
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${take(replace('${resourcePrefix}${environmentName}', '-', ''), 10)}-${take(uniqueString(resourceGroup().id, resourcePrefix, environmentName, '2025'), 10)}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: config.keyVaultSku
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: true
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
  tags: {
    Environment: environmentName
    Purpose: 'Secure storage of secrets, keys, and certificates'
  }
}

// Container Registry removed - not supported in Azure for Students subscription
// Azure ML will use local training without container registry dependency

// RBAC assignments for managed identity
resource searchIndexDataContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: searchService
  name: guid(searchService.id, managedIdentity.id, 'Search Index Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
    )
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource searchServiceContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: searchService
  name: guid(searchService.id, managedIdentity.id, 'Search Service Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
    )
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource storageDataContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, managedIdentity.id, 'Storage Blob Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
    )
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource keyVaultSecretsOfficer 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, managedIdentity.id, 'Key Vault Secrets Officer')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
    )
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ACR RBAC removed - Container Registry not supported in Azure for Students

// RBAC for current user (for development)
resource userKeyVaultSecretsOfficer 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: keyVault
  name: guid(keyVault.id, principalId, 'Key Vault Secrets Officer')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
    )
    principalId: principalId
    principalType: 'User'
  }
}

// User search permissions for local development
resource userSearchIndexDataContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: searchService
  name: guid(searchService.id, principalId, 'Search Index Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
    )
    principalId: principalId
    principalType: 'User'
  }
}

resource userSearchServiceContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: searchService
  name: guid(searchService.id, principalId, 'Search Service Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
    )
    principalId: principalId
    principalType: 'User'
  }
}

// User storage permissions for local development
resource userStorageDataContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: storageAccount
  name: guid(storageAccount.id, principalId, 'Storage Blob Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
    )
    principalId: principalId
    principalType: 'User'
  }
}

// Outputs
output managedIdentityId string = managedIdentity.id
output managedIdentityClientId string = managedIdentity.properties.clientId
output managedIdentityPrincipalId string = managedIdentity.properties.principalId

output searchServiceName string = searchService.name
output searchEndpoint string = 'https://${searchService.name}.search.windows.net/'

output storageAccountName string = storageAccount.name
output storageAccountEndpoint string = storageAccount.properties.primaryEndpoints.blob

output keyVaultName string = keyVault.name
output keyVaultEndpoint string = keyVault.properties.vaultUri

output appInsightsName string = appInsights.name
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output appInsightsInstrumentationKey string = appInsights.properties.InstrumentationKey

output logAnalyticsWorkspaceId string = logAnalytics.properties.customerId
output logAnalyticsWorkspaceName string = logAnalytics.name

// Container Registry outputs removed - not supported in Azure for Students
output containerRegistryName string = 'not-available-in-student-subscription'
output containerRegistryEndpoint string = 'not-available-in-student-subscription'
