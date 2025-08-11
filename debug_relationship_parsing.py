import asyncio
import sys

async def debug_relationship_parsing():
    print('🔬 Testing relationship extraction parsing...')
    
    content = '''Azure AI Language service provides natural language processing capabilities.
    It includes sentiment analysis, entity recognition, and key phrase extraction.'''
    
    from agents.core.universal_deps import get_universal_deps
    from agents.core.prompt_cache import get_or_generate_auto_prompts
    from agents.knowledge_extraction.agent import run_knowledge_extraction
    
    try:
        # First get entities (which we know works)
        result = await run_knowledge_extraction(
            content=content,
            use_domain_analysis=True,
            verbose=False
        )
        
        print(f'✅ Entities extracted: {len(result.entities)}')
        
        # Now test relationship extraction specifically
        deps = await get_universal_deps()
        cached_prompts = await get_or_generate_auto_prompts(content=content, verbose=False)
        
        # Get the relationship extraction prompt
        relationship_prompt = cached_prompts.extraction_prompts.get('relationship_extraction', '')
        print(f'\n📝 Relationship extraction prompt (first 200 chars):')
        print(relationship_prompt[:200] + '...')
        
        # Call Azure OpenAI directly for relationship extraction
        openai_client = deps.openai_client
        print(f'\n🔄 Calling Azure OpenAI for relationships...')
        
        response = await openai_client.get_completion(
            relationship_prompt,
            max_tokens=800,
            temperature=0.2
        )
        
        print(f'\n📄 Relationship Response:')
        print(f'Length: {len(response)} chars')
        print('Content:')
        print('-' * 80)
        print(response)
        print('-' * 80)
        
        # Try to parse the response
        import json
        import re
        
        json_text = None
        
        # Try array first
        bracket_start = response.find('[')
        if bracket_start != -1:
            bracket_count = 0
            for i, char in enumerate(response[bracket_start:], bracket_start):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_text = response[bracket_start:i+1]
                        break
        
        if json_text:
            print(f'\n✅ Found JSON array: {json_text}')
            try:
                result_data = json.loads(json_text)
                print(f'📊 Parsed successfully: {len(result_data)} relationships')
                for i, rel in enumerate(result_data[:2]):
                    print(f'   {i+1}. {rel}')
            except json.JSONDecodeError as e:
                print(f'❌ JSON parsing failed: {e}')
        else:
            print(f'❌ No JSON array found - checking for object...')
            
            # Try object
            brace_start = response.find('{')
            if brace_start != -1:
                print(f'Found JSON object at position {brace_start}')
                
        return True
        
    except Exception as e:
        print(f'❌ Debug failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(debug_relationship_parsing())
    print(f'Debug result: {result}')