// AI Services Module: Azure OpenAI with model deployments
param environmentName string
param location string
param principalId string
param resourcePrefix string

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

// GPT-4 Model Deployment for text generation and reasoning
resource gpt4Deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openaiAccount
  name: 'gpt-4'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-05-13'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    sku: {
      name: 'Standard'
      capacity: config.gpt4Capacity
    }
  }
}

// GPT-4 Turbo for enhanced performance (production)
resource gpt4TurboDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (environmentName == 'production') {
  parent: openaiAccount
  name: 'gpt-4-turbo'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4'
      version: '1106-Preview'
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
    sku: {
      name: 'Standard'
      capacity: 20
    }
  }
  dependsOn: [gpt4Deployment]
}

// Text Embedding Model for vector search
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
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
      name: 'Standard'
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
      name: 'Standard'
      capacity: 30
    }
  }
  dependsOn: [embeddingDeployment]
}

// RBAC for managed identity access
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

output deploymentName string = gpt4Deployment.name
output embeddingDeploymentName string = embeddingDeployment.name
output gpt4TurboDeploymentName string = environmentName == 'production' ? gpt4TurboDeployment.name : ''
output embedding3LargeDeploymentName string = environmentName == 'production' ? embedding3LargeDeployment.name : ''

output openaiLocation string = config.location
output openaiResourceGroup string = resourceGroup().name