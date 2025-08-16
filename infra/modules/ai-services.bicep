// AI Services Module: Azure OpenAI with model deployments
param environmentName string
param location string
param principalId string
param resourcePrefix string
param managedIdentityPrincipalId string

// Single configuration - COST OPTIMIZED (MINI MODEL ONLY)
var config = {
  miniCapacity: 1         // MINIMAL: Single capacity unit for mini model
  embeddingCapacity: 1    // MINIMAL: Single capacity unit for embeddings
  location: 'eastus'      // AZURE FOR STUDENTS: Try East US for better availability
}

var deploymentLocation = location

// Azure AI Foundry Service - Available in Azure for Students
resource openaiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: 'aif-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: deploymentLocation
  kind: 'AIServices'  // Azure AI Foundry - multi-service
  sku: {
    name: 'S0'  // Standard tier for AI Services
  }
  properties: {
    customSubDomainName: 'aif-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
    publicNetworkAccess: 'Enabled'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Azure AI Foundry multi-service - Azure for Students'
  }
}

// GPT-4o-Mini Model Deployment - Most cost-effective chat model
resource gpt4oMiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openaiAccount
  name: 'gpt-4o-mini'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o-mini'
      version: '2024-07-18'
    }
    raiPolicyName: 'Microsoft.Default'
    scaleSettings: {
      scaleType: 'Standard'
      capacity: config.miniCapacity  // Use minimal capacity
    }
  }
}


// Text Embedding Model - Keep ada-002 as it's cost-effective
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openaiAccount
  name: 'text-embedding-ada-002'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
    }
    raiPolicyName: 'Microsoft.Default'
    scaleSettings: {
      scaleType: 'Standard'
      capacity: config.embeddingCapacity  // Use minimal capacity
    }
  }
  dependsOn: [
    gpt4oMiniDeployment  // Deploy sequentially to avoid conflicts
  ]
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

output deploymentName string = 'gpt-4o-mini'  // Changed to mini model for cost optimization
output embeddingDeploymentName string = 'text-embedding-ada-002'
output gpt4TurboDeploymentName string = ''  // Not deployed for cost savings
output embedding3LargeDeploymentName string = ''  // Not deployed for cost savings

output openaiLocation string = config.location
output openaiResourceGroup string = resourceGroup().name