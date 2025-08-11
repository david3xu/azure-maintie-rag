import asyncio
import sys
import json
import re

async def debug_extraction_response():
    print('🔬 Testing Azure OpenAI response parsing...')
    
    # Test with a simple sample
    content = '''Azure AI Language service provides natural language processing capabilities.'''
    
    from agents.core.universal_deps import get_universal_deps
    from agents.core.prompt_cache import get_or_generate_auto_prompts
    
    try:
        # Get cached prompts
        cached_prompts = await get_or_generate_auto_prompts(content=content, verbose=False)
        
        # Get the entity extraction prompt
        entity_prompt = cached_prompts.extraction_prompts.get('entity_extraction', '')
        print(f'\n📝 Entity extraction prompt (first 300 chars):')
        print(entity_prompt[:300] + '...')
        
        # Call Azure OpenAI directly to see raw response
        deps = await get_universal_deps()
        openai_client = deps.openai_client
        
        print(f'\n🔄 Calling Azure OpenAI directly...')
        response = await openai_client.get_completion(
            entity_prompt,
            max_tokens=1200,
            temperature=0.3
        )
        
        print(f'\n📄 Azure OpenAI Response:')
        print(f'Length: {len(response)} chars')
        print('Content:')
        print('-' * 80)
        print(response)
        print('-' * 80)
        
        # Try to find JSON in response
        json_text = None
        
        # Strategy 1: Find first complete JSON object
        brace_count = 0
        start_index = response.find('{')
        if start_index != -1:
            for i, char in enumerate(response[start_index:], start_index):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_text = response[start_index:i+1]
                        break
        
        if json_text:
            print(f'\n✅ Found JSON: {json_text}')
            try:
                result_data = json.loads(json_text)
                entities = result_data.get('entities', [])
                print(f'📊 Entities in JSON: {len(entities)}')
                if entities:
                    for i, e in enumerate(entities[:3]):
                        print(f'   {i+1}. {e}')
                else:
                    print('   No entities found in JSON')
            except json.JSONDecodeError as e:
                print(f'❌ JSON parsing failed: {e}')
        else:
            print(f'❌ No JSON found in response')
            print('Checking for other patterns...')
            
            # Look for array patterns
            if '[' in response and ']' in response:
                print('Found potential array pattern')
            
            # Look for common LLM response patterns
            if 'entities:' in response.lower():
                print('Found "entities:" text pattern')
                
        return True
        
    except Exception as e:
        print(f'❌ Debug failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(debug_extraction_response())
    print(f'Debug result: {result}')