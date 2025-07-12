from openai import AzureOpenAI
from config.settings import settings
import os

def test_azure_embedding_api():
    """Test actual Azure OpenAI embedding API"""
    print("🔄 Testing Azure OpenAI Embedding API...")

    try:
        # Initialize Azure OpenAI client (real implementation)
        client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version=settings.embedding_api_version,
            azure_endpoint=settings.embedding_api_base
        )

        # Test embedding call (small batch)
        response = client.embeddings.create(
            model=settings.embedding_deployment_name,
            input=["test maintenance query"]
        )

        embedding = response.data[0].embedding
        print(f"✅ Azure embedding API working")
        print(f"📊 Embedding dimension: {len(embedding)}")
        print(f"📊 Model: {settings.embedding_deployment_name}")

        return True

    except Exception as e:
        print(f"❌ Azure embedding API error: {e}")
        return False

def test_azure_chat_api():
    """Test actual Azure OpenAI chat API"""
    print("🔄 Testing Azure OpenAI Chat API...")

    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            azure_endpoint=settings.openai_api_base
        )

        # Test chat completion
        response = client.chat.completions.create(
            model=settings.openai_deployment_name,
            messages=[
                {"role": "user", "content": "What is pump maintenance?"}
            ],
            max_tokens=50
        )

        print(f"✅ Azure chat API working")
        print(f"📊 Response: {response.choices[0].message.content[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Azure chat API error: {e}")
        return False

if __name__ == "__main__":
    test_azure_embedding_api()
    test_azure_chat_api()
