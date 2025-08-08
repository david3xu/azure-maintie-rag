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
  prod: {
    gpt4Capacity: 50
    embeddingCapacity: 60
    location: 'westus'
  }
}

var config = environmentConfig[environmentName]
var deploymentLocation = location

// Azure OpenAI Service (existing)
resource openaiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' existing = {
  name: 'oai-maintie-rag-prod-fymhwfec3ra2w'
}

// GPT-4o Model Deployment (existing)
resource gpt4Deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' existing = {
  parent: openaiAccount
  name: 'gpt-4o'
}


// Text Embedding Model (existing)
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' existing = {
  parent: openaiAccount
  name: 'text-embedding-ada-002'
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