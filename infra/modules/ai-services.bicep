// AI Services Module: Azure OpenAI with model deployments
param environmentName string
param location string
param principalId string
param resourcePrefix string
param managedIdentityPrincipalId string

// Environment-specific model configurations
var environmentConfig = {
  development: {
    gpt4Capacity: 10
    embeddingCapacity: 15
    location: 'westus' // Use different region for better availability
  }
  staging: {
    gpt4Capacity: 20
    embeddingCapacity: 30
    location: 'westus'
  }
  production: {
    gpt4Capacity: 50
    embeddingCapacity: 60
    location: 'westus'
  }
}

var config = environmentConfig[environmentName]
var deploymentLocation = location

// Azure OpenAI Service
resource openaiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: 'oai-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: config.location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: '${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
  tags: {
    Environment: environmentName
    Purpose: 'AI text processing, embeddings, and completions for Universal RAG'
  }
}

// GPT-4o Model Deployment for text generation and reasoning
resource gpt4Deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (true) {
  parent: openaiAccount
  name: 'gpt-4o'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-08-06'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    sku: {
      name: 'GlobalStandard'
      capacity: config.gpt4Capacity
    }
  }
}

// GPT-4o Mini for enhanced performance (production)
resource gpt4TurboDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (environmentName == 'production') {
  parent: openaiAccount
  name: 'gpt-4o-mini'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o-mini'
      version: '2024-07-18'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    sku: {
      name: 'GlobalStandard'
      capacity: 20
    }
  }
  dependsOn: [gpt4Deployment]
}

// Text Embedding Model for vector search
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (true) {
  parent: openaiAccount
  name: 'text-embedding-ada-002'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    sku: {
      name: 'GlobalStandard'
      capacity: config.embeddingCapacity
    }
  }
  dependsOn: [gpt4Deployment]
}

// Text Embedding 3 Large (for production environments)
resource embedding3LargeDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (environmentName == 'production') {
  parent: openaiAccount
  name: 'text-embedding-3-large'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-large'
      version: '1'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    sku: {
      name: 'GlobalStandard'
      capacity: 30
    }
  }
  dependsOn: [embeddingDeployment]
}

// RBAC for managed identity access
resource managedIdentityOpenaiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: openaiAccount
  name: guid(openaiAccount.id, managedIdentityPrincipalId, 'Cognitive Services OpenAI User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: managedIdentityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// RBAC for user access (development)
resource openaiUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: openaiAccount
  name: guid(openaiAccount.id, principalId, 'Cognitive Services OpenAI User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: principalId
    principalType: 'User'
  }
}

// Monitor deployment status and quotas
resource diagnosticSettings 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  scope: openaiAccount
  name: 'openai-diagnostics'
  properties: {
    workspaceId: resourceId('Microsoft.OperationalInsights/workspaces', 'log-${resourcePrefix}-${environmentName}')
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
        retentionPolicy: {
          enabled: false
          days: 0
        }
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
        retentionPolicy: {
          enabled: false
          days: 0
        }
      }
    ]
  }
}

// Outputs
output openaiAccountName string = openaiAccount.name
output openaiEndpoint string = openaiAccount.properties.endpoint
output openaiResourceId string = openaiAccount.id

output deploymentName string = 'gpt-4o'
output embeddingDeploymentName string = 'text-embedding-ada-002'
output gpt4TurboDeploymentName string = ''
output embedding3LargeDeploymentName string = ''

output openaiLocation string = config.location
output openaiResourceGroup string = resourceGroup().name