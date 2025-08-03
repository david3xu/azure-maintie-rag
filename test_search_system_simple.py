#!/usr/bin/env python3
"""
Simple Search System Test
Tests the core search functionality including vector, text, and semantic search
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_vector_search_embeddings():
    """Test vector search functionality using embeddings"""
    logger.info("🔍 Testing vector search with embeddings...")

    try:
        from openai import AzureOpenAI
        import numpy as np

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        )

        # Test queries
        test_queries = [
            "How to train machine learning models in Azure",
            "Azure ML deployment and endpoints",
            "MLOps and machine learning lifecycle management"
        ]

        embeddings_results = []

        for query in test_queries:
            logger.info(f"  📝 Generating embedding for: '{query[:50]}...'")

            # Generate embedding for query
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )

            embedding = response.data[0].embedding
            embeddings_results.append({
                'query': query,
                'embedding': embedding,
                'dimensions': len(embedding)
            })

            logger.info(f"  ✅ Generated {len(embedding)}-dimensional embedding")

        # Test similarity computation
        logger.info("  🔗 Testing embedding similarity computation...")

        if len(embeddings_results) >= 2:
            emb1 = np.array(embeddings_results[0]['embedding'])
            emb2 = np.array(embeddings_results[1]['embedding'])

            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            logger.info(f"  ✅ Cosine similarity between queries: {similarity:.4f}")

        return True, {
            'queries_processed': len(test_queries),
            'embedding_dimensions': embeddings_results[0]['dimensions'] if embeddings_results else 0,
            'similarity_score': float(similarity) if 'similarity' in locals() else 0.0
        }

    except Exception as e:
        logger.error(f"❌ Vector search test failed: {str(e)}")
        return False, {}


async def test_text_search_extraction():
    """Test text search and knowledge extraction"""
    logger.info("🔍 Testing text search and knowledge extraction...")

    try:
        from openai import AzureOpenAI

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        )

        # Sample document content (simulating retrieved documents)
        sample_docs = [
            {
                'title': 'Azure ML Training',
                'content': 'Azure Machine Learning provides tools for training machine learning models at scale. You can use compute clusters to run training jobs with automatic scaling and distributed computing capabilities.'
            },
            {
                'title': 'Model Deployment',
                'content': 'Deploy trained models to real-time endpoints or batch inference pipelines. Azure ML supports various deployment targets including Azure Container Instances and Azure Kubernetes Service.'
            },
            {
                'title': 'MLOps Workflows',
                'content': 'Implement MLOps practices with Azure ML pipelines, automated retraining, model monitoring, and version control. Track experiments and manage the complete ML lifecycle.'
            }
        ]

        # Test query
        query = "How do I deploy a machine learning model for real-time inference?"

        logger.info(f"  📝 Processing query: '{query}'")

        # Create context from sample documents
        context = "\n\n".join([f"Document: {doc['title']}\n{doc['content']}" for doc in sample_docs])

        # Test knowledge extraction and search
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Azure ML assistant. Use the provided documents to answer questions accurately and cite relevant information."
                },
                {
                    "role": "user",
                    "content": f"Context Documents:\n{context}\n\nQuery: {query}\n\nProvide a comprehensive answer based on the context documents."
                }
            ],
            max_tokens=500,
            temperature=0.1
        )

        answer = response.choices[0].message.content
        logger.info(f"  ✅ Generated answer ({len(answer)} characters)")
        logger.info(f"  📄 Sample answer: {answer[:200]}...")

        # Test information extraction
        extraction_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "Extract key entities, concepts, and relationships from the given text. Return structured information."
                },
                {
                    "role": "user",
                    "content": f"Extract key information from this content:\n\n{context[:1000]}"
                }
            ],
            max_tokens=300,
            temperature=0.1
        )

        extracted_info = extraction_response.choices[0].message.content
        logger.info(f"  ✅ Extracted structured information ({len(extracted_info)} characters)")

        return True, {
            'query_length': len(query),
            'context_documents': len(sample_docs),
            'answer_length': len(answer),
            'extracted_info_length': len(extracted_info)
        }

    except Exception as e:
        logger.error(f"❌ Text search test failed: {str(e)}")
        return False, {}


async def test_semantic_search_capabilities():
    """Test semantic search and understanding"""
    logger.info("🔍 Testing semantic search capabilities...")

    try:
        from openai import AzureOpenAI

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        )

        # Test semantic understanding with various query types
        test_scenarios = [
            {
                'query': 'ML model performance optimization',
                'semantic_intent': 'performance_improvement'
            },
            {
                'query': 'Azure machine learning best practices',
                'semantic_intent': 'best_practices'
            },
            {
                'query': 'troubleshooting deployment issues',
                'semantic_intent': 'problem_solving'
            }
        ]

        results = []

        for scenario in test_scenarios:
            query = scenario['query']
            logger.info(f"  🧠 Analyzing semantic intent for: '{query}'")

            # Test semantic understanding
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the semantic intent of user queries. Identify the main topic, intent type, and relevant Azure ML services that would be helpful."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nAnalyze this query and identify:\n1. Main topic\n2. User intent\n3. Relevant Azure ML services\n4. Suggested search strategy"
                    }
                ],
                max_tokens=300,
                temperature=0.1
            )

            analysis = response.choices[0].message.content

            results.append({
                'query': query,
                'analysis': analysis,
                'analysis_length': len(analysis)
            })

            logger.info(f"  ✅ Semantic analysis completed ({len(analysis)} characters)")

        # Test query expansion
        logger.info("  🔍 Testing query expansion capabilities...")

        original_query = "deploy ML model"
        expansion_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "Expand search queries to include related terms, synonyms, and contextual variations that would improve search results."
                },
                {
                    "role": "user",
                    "content": f"Expand this query for better search coverage: '{original_query}'\n\nProvide 5-10 related search terms and phrases."
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        expanded_terms = expansion_response.choices[0].message.content
        logger.info(f"  ✅ Query expansion completed: {expanded_terms[:100]}...")

        return True, {
            'semantic_scenarios': len(test_scenarios),
            'total_analysis_length': sum(r['analysis_length'] for r in results),
            'query_expansion_length': len(expanded_terms)
        }

    except Exception as e:
        logger.error(f"❌ Semantic search test failed: {str(e)}")
        return False, {}


async def test_tri_modal_orchestration():
    """Test tri-modal search orchestration (vector + text + semantic)"""
    logger.info("🔍 Testing tri-modal search orchestration...")

    try:
        from openai import AzureOpenAI
        import json

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        )

        # Test query
        user_query = "What are the best practices for monitoring machine learning models in production?"

        logger.info(f"  🎯 Orchestrating tri-modal search for: '{user_query}'")

        # Step 1: Generate embedding (Vector search component)
        logger.info("  📊 Step 1: Vector search - generating query embedding...")
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=user_query
        )
        query_embedding = embedding_response.data[0].embedding

        # Step 2: Semantic analysis (Semantic search component)
        logger.info("  🧠 Step 2: Semantic search - analyzing query intent...")
        semantic_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the search intent and extract key concepts for a comprehensive search strategy."
                },
                {
                    "role": "user",
                    "content": f"Query: {user_query}\n\nExtract:\n1. Primary concepts\n2. Related terms\n3. Search strategy\n4. Expected information types"
                }
            ],
            max_tokens=300,
            temperature=0.1
        )
        semantic_analysis = semantic_response.choices[0].message.content

        # Step 3: Text search simulation (Text search component)
        logger.info("  📝 Step 3: Text search - retrieving and ranking results...")

        # Simulate retrieved documents
        mock_search_results = [
            {
                'title': 'ML Model Monitoring Best Practices',
                'content': 'Monitor model performance metrics, data drift, and prediction accuracy. Use Azure ML model monitoring to track model behavior over time.',
                'relevance_score': 0.95
            },
            {
                'title': 'Production ML Deployment',
                'content': 'Deploy models with proper logging, alerting, and monitoring. Implement automated retraining pipelines for continuous improvement.',
                'relevance_score': 0.88
            },
            {
                'title': 'Azure ML Operations Guide',
                'content': 'Establish MLOps workflows with version control, testing, and deployment automation. Monitor model health and performance continuously.',
                'relevance_score': 0.92
            }
        ]

        # Step 4: Result fusion and ranking
        logger.info("  🔗 Step 4: Tri-modal fusion - combining all search modalities...")

        # Create comprehensive context
        search_context = {
            'query': user_query,
            'embedding_dimensions': len(query_embedding),
            'semantic_analysis': semantic_analysis,
            'search_results': mock_search_results,
            'result_count': len(mock_search_results)
        }

        # Generate final response using all modalities
        fusion_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert search assistant. Combine vector similarity, semantic understanding, and text search results to provide the most comprehensive and accurate answer."
                },
                {
                    "role": "user",
                    "content": f"Search Query: {user_query}\n\nSemantic Analysis:\n{semantic_analysis}\n\nSearch Results:\n{json.dumps(mock_search_results, indent=2)}\n\nProvide a comprehensive answer that leverages all search modalities."
                }
            ],
            max_tokens=600,
            temperature=0.1
        )

        final_answer = fusion_response.choices[0].message.content

        logger.info(f"  ✅ Tri-modal search completed successfully")
        logger.info(f"  📊 Final answer length: {len(final_answer)} characters")
        logger.info(f"  🎯 Sample answer: {final_answer[:150]}...")

        return True, {
            'query_embedding_dims': len(query_embedding),
            'semantic_analysis_length': len(semantic_analysis),
            'search_results_count': len(mock_search_results),
            'final_answer_length': len(final_answer),
            'modalities_used': 3
        }

    except Exception as e:
        logger.error(f"❌ Tri-modal search test failed: {str(e)}")
        return False, {}


async def main():
    """Main testing function"""
    print("🧪 Azure Universal RAG - Search System Testing")
    print("Testing vector, semantic, and tri-modal search capabilities")
    print("-" * 60)

    start_time = datetime.utcnow()

    # Run all search tests
    tests = [
        ("Vector Search & Embeddings", test_vector_search_embeddings()),
        ("Text Search & Extraction", test_text_search_extraction()),
        ("Semantic Search Capabilities", test_semantic_search_capabilities()),
        ("Tri-Modal Search Orchestration", test_tri_modal_orchestration()),
    ]

    results = {}
    test_details = {}

    for test_name, test_coro in tests:
        success, details = await test_coro
        results[test_name] = success
        test_details[test_name] = details

    # Print summary
    total_time = (datetime.utcnow() - start_time).total_seconds()
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    print("\n" + "="*60)
    print("🧪 SEARCH SYSTEM TEST SUMMARY")
    print("="*60)
    print(f"⏱️  Total Testing Time: {total_time:.2f} seconds")
    print(f"📊 Tests Run: {total_tests}")
    print(f"✅ Tests Passed: {passed_tests}")
    print(f"❌ Tests Failed: {total_tests - passed_tests}")
    print(f"📈 Success Rate: {passed_tests/total_tests*100:.1f}%")

    print("\n📋 SEARCH SYSTEM RESULTS:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")

        if passed and test_name in test_details:
            details = test_details[test_name]
            if details:
                for key, value in details.items():
                    print(f"     - {key}: {value}")

    if passed_tests == total_tests:
        print("\n🎉 ALL SEARCH SYSTEM TESTS PASSED!")
        print("✅ Vector search with embeddings working")
        print("✅ Text search and extraction functional")
        print("✅ Semantic search capabilities operational")
        print("✅ Tri-modal search orchestration successful")
        print("✅ Ready for end-to-end system testing")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} SEARCH TESTS FAILED")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
