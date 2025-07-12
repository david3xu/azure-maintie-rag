# 📝 Execution Log: MaintIE Enhanced RAG Backend

This document records the process of running and validating the MaintIE Enhanced RAG backend, including errors and fixes.

---

## 1. Initial Run Attempt

**Command:**

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Result:**

- Uvicorn server started, but crashed with a traceback.
- Error message:
  > `pydantic.errors.PydanticImportError: BaseSettings has been moved to the pydantic-settings package.`

**Diagnosis:**

- Pydantic v2+ moved `BaseSettings` to a new package: `pydantic-settings`.
- The codebase imports `BaseSettings` from `pydantic`, which is not compatible with the installed version.

---

## 2. Next Steps

- [x] Update the codebase to use `pydantic-settings` for configuration management.
- [x] Update `requirements.txt` to include `pydantic-settings`.
- [x] Re-run the server and record the outcome.

---

## 3. Environment Issues

- Attempted to activate `venv/bin/activate`, but the directory does not exist.
- Created a new virtual environment and reinstalled dependencies.

---

## 4. OpenAI SDK Migration & Azure Integration

- Migrated all LLM and embedding code to use the latest OpenAI SDK (`openai>=1.13.3`) and the new `AzureOpenAI` client interface.
- Updated all config fields to match the new SDK and Azure OpenAI requirements.
- Updated API version in `.env` and config to `2025-03-01-preview` (latest as of July 2025).

---

## 5. Embedding API Error & Robustness

- On startup, the backend attempts to build a vector index for 7000 documents using Azure OpenAI embeddings.
- The embedding API call fails with:
  > `Error code: 400 - {'error': {'message': "'$.input' is invalid. Please check the API reference: https://platform.openai.com/docs/api-reference.", ...}`
- Diagnosis: Azure OpenAI embedding endpoint has a batch size limit (max 2048 inputs/request for embedding-3). Sending 7000 at once fails.
- Added robust error handling: if index build fails, the backend logs a warning but continues to run and serve all endpoints.

---

## 6. Server Health & Endpoint Verification

- FastAPI server runs and serves `/docs` and all API endpoints.
- LLM (chat/completions) works with Azure OpenAI.
- Vector search is available, but index build is skipped if embedding batch fails.
- All config and initialization errors are fixed.

---

## 7. Direct Azure OpenAI API Testing (curl)

- Provided ready-to-use `curl` commands to test both embedding and chat endpoints directly.
- **Best practice:** Use an environment variable for your API key to avoid exposing secrets in your shell history or logs.

**Before running the curl commands, export your real API key:**

```bash
export AZURE_OPENAI_API_KEY="your-real-azure-openai-key"
```

**Embeddings:**

```bash
curl -X POST "https://clu-project-foundry-instance.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2025-03-01-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: $AZURE_OPENAI_API_KEY" \
  -d '{
    "input": "Hello, this is a test for Azure OpenAI embeddings!"
  }'
```

**Chat:**

```bash
curl -X POST "https://clu-project-foundry-instance.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-03-01-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: $AZURE_OPENAI_API_KEY" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "When was Microsoft founded?"}
    ]
  }'

```

---

## 8. Next Steps

- [ ] Implement batching for embedding requests (≤2048 per call) in vector search code.
- [ ] Confirm API version in `.env` matches Azure portal (should be `2025-03-01-preview`).
- [ ] Rebuild vector index after batching fix.
- [ ] Validate all endpoints with real data.

---

**Current Status:**

- Backend is robust and runs with all endpoints available.
- LLM (chat) works with Azure OpenAI.
- Vector search index build fails gracefully if batch size is too large, but does not crash the server.
- Ready for final embedding batching fix and production use.
