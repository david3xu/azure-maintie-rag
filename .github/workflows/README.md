# CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing and deployment.

## Workflows

### 1. CI Workflow (`ci.yml`)
- **Trigger**: Push to main/develop/feature branches, PRs
- **Purpose**: Run tests and validate code quality
- **Steps**:
  - Install Python dependencies
  - Run syntax checks
  - Execute unit tests
  - Run basic integration tests (no Azure credentials required)

### 2. CD Workflow (`cd.yml`)
- **Trigger**: After successful CI completion
- **Purpose**: Deploy to staging/production environments
- **Environments**: staging (develop branch), production (main branch)

### 3. Docker Build Workflow (`docker.yml`)
- **Trigger**: Push to main/develop branches
- **Purpose**: Build and push container images
- **Features**: Security scanning with Trivy

## Required Secrets

### Azure Deployment Secrets
```
AZURE_CREDENTIALS           # Azure service principal credentials (JSON)
AZURE_REGISTRY_URL         # Azure Container Registry URL
REGISTRY_USERNAME          # Registry username
REGISTRY_PASSWORD          # Registry password/token
```

### Environment URLs
```
STAGING_BACKEND_URL        # Staging backend URL for health checks
PRODUCTION_BACKEND_URL     # Production backend URL for health checks  
```

### Azure OpenAI Secrets (Optional - for full integration tests)
```
OPENAI_API_KEY            # Azure OpenAI API key
OPENAI_API_BASE           # Azure OpenAI endpoint
OPENAI_API_VERSION        # API version
OPENAI_DEPLOYMENT_NAME    # GPT model deployment name
OPENAI_MODEL              # Model name
EMBEDDING_MODEL           # Embedding model name
EMBEDDING_DEPLOYMENT_NAME # Embedding deployment name
EMBEDDING_API_BASE        # Embedding endpoint
EMBEDDING_API_VERSION     # Embedding API version
```

## Azure Credentials Setup

### 1. Create Service Principal
```bash
az ad sp create-for-rbac \
  --name "github-actions-maintie-rag" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} \
  --sdk-auth
```

### 2. Add Required Permissions
The service principal needs permissions for:
- Azure Container Registry (push/pull images)
- Azure Container Apps (update deployments)
- Resource Groups (read resources)

### 3. Format AZURE_CREDENTIALS Secret
```json
{
  "clientId": "<client-id>",
  "clientSecret": "<client-secret>",
  "subscriptionId": "<subscription-id>",
  "tenantId": "<tenant-id>"
}
```

## Container Registry Setup

### Azure Container Registry
```bash
# Create ACR
az acr create --resource-group myResourceGroup --name myRegistry --sku Basic

# Get login server
az acr show --name myRegistry --query loginServer --output table

# Set secrets in GitHub:
# AZURE_REGISTRY_URL: myregistry.azurecr.io
# REGISTRY_USERNAME: myregistry
# REGISTRY_PASSWORD: <admin-password-or-token>
```

### GitHub Container Registry (Alternative)
```bash
# Secrets for GHCR:
# REGISTRY_USERNAME: <github-username>
# REGISTRY_PASSWORD: <github-personal-access-token>
```

## Workflow Status

- ✅ **CI**: Basic syntax and unit testing (no Azure dependencies)
- ✅ **Docker**: Container building and security scanning
- ⚠️ **CD**: Requires Azure credentials and resource setup
- ⚠️ **Integration Tests**: Requires Azure OpenAI credentials

## Troubleshooting

### Common Issues

1. **Tests fail on missing dependencies**
   - Check `backend/requirements.txt` is up to date
   - Ensure test structure matches workflow paths

2. **Docker build fails**
   - Verify Dockerfile is in `backend/` directory
   - Check for missing files referenced in Dockerfile

3. **Deployment fails**
   - Verify Azure credentials have correct permissions
   - Check resource names match Azure infrastructure
   - Ensure Container Apps exist and are accessible

4. **Health checks fail**
   - Verify backend URLs are correct
   - Check if services are properly deployed
   - Ensure health endpoint returns 200 status

## Local Testing

### Test CI workflow locally
```bash
cd backend
python -m pip install -r requirements.txt
python -m py_compile api/main.py
pytest tests/unit/ -v
pytest tests/integration/test_imports.py tests/integration/test_syntax.py -v
```

### Test Docker build locally
```bash
cd backend
docker build -t maintie-rag-backend .
docker run -p 8000:8000 maintie-rag-backend
curl http://localhost:8000/api/v1/health
```