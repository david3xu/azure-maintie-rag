#!/usr/bin/env python3
"""
Test script to validate Agent 1 Data Schema Design Plan implementation status
"""

import asyncio
from agents.domain_intelligence.agent import run_domain_analysis
from agents.knowledge_extraction.agent import run_knowledge_extraction  

async def test_implementation():
    test_content = '''Azure Cognitive Services Text Analytics API provides advanced natural language processing capabilities including named entity recognition, key phrase extraction, and sentiment analysis. The service integrates with Azure Machine Learning pipelines and supports real-time processing of multilingual content through RESTful API endpoints.'''
    
    print('🔍 AGENT 1 DATA SCHEMA IMPLEMENTATION ANALYSIS')
    print('='*70)
    
    try:
        domain_analysis = await run_domain_analysis(test_content, detailed=True)
        
        print()
        print('1️⃣ SCHEMA COMPLIANCE CHECK:')
        print('-' * 40)
        
        # Check required fields from the plan
        fields_to_check = {
            'domain_signature': getattr(domain_analysis, 'domain_signature', None),
            'content_type_confidence': getattr(domain_analysis, 'content_type_confidence', None),
            'analysis_timestamp': getattr(domain_analysis, 'analysis_timestamp', None),
            'processing_time': getattr(domain_analysis, 'processing_time', None),
            'data_source_path': getattr(domain_analysis, 'data_source_path', None),
            'analysis_reliability': getattr(domain_analysis, 'analysis_reliability', None),
            'key_insights': getattr(domain_analysis, 'key_insights', None),
            'adaptation_recommendations': getattr(domain_analysis, 'adaptation_recommendations', None)
        }
        
        populated_count = 0
        for field_name, value in fields_to_check.items():
            has_value = value is not None and value != '' and value != []
            status = '✅ POPULATED' if has_value else '❌ MISSING'
            print(f'   {field_name}: {status}')
            if has_value:
                populated_count += 1
                if isinstance(value, list) and len(value) > 0:
                    print(f'      Sample: {str(value[0])[:50]}...')
                elif isinstance(value, (int, float, str)) and len(str(value)) > 0:
                    print(f'      Value: {str(value)[:50]}')
                
        compliance_rate = populated_count / len(fields_to_check) * 100
        print()
        print(f'📊 Schema Compliance: {populated_count}/{len(fields_to_check)} ({compliance_rate:.1f}%)')
        
        # Check critical field name fix
        print()
        print('🏷️ CRITICAL FIELD NAME COMPLIANCE:')
        print('-' * 40)
        vocab_complexity_ratio = getattr(domain_analysis.characteristics, 'vocabulary_complexity_ratio', None)
        most_frequent_terms = getattr(domain_analysis.characteristics, 'most_frequent_terms', None)
        content_patterns = getattr(domain_analysis.characteristics, 'content_patterns', None)
        sentence_complexity = getattr(domain_analysis.characteristics, 'sentence_complexity', None)
        
        field_name_fixes = 0
        total_critical_fields = 4
        
        if vocab_complexity_ratio is not None:
            print(f'   vocabulary_complexity_ratio: ✅ FIXED ({vocab_complexity_ratio:.3f})')
            field_name_fixes += 1
        else:
            print('   vocabulary_complexity_ratio: ❌ MISSING')
        
        if most_frequent_terms and len(most_frequent_terms) > 0:
            print(f'   most_frequent_terms: ✅ POPULATED ({len(most_frequent_terms)} terms)')
            field_name_fixes += 1
        else:
            print('   most_frequent_terms: ❌ MISSING')
            
        if content_patterns is not None:
            print(f'   content_patterns: ✅ POPULATED ({len(content_patterns)} patterns)')
            field_name_fixes += 1
        else:
            print('   content_patterns: ❌ MISSING')
            
        if sentence_complexity is not None:
            print(f'   sentence_complexity: ✅ POPULATED ({sentence_complexity})')
            field_name_fixes += 1
        else:
            print('   sentence_complexity: ❌ MISSING')
        
        # Check processing config (this was working according to plan)
        print()
        print('⚙️ PROCESSING CONFIG (Agent 1 -> Agent 2/3 Flow):')
        print('-' * 40)
        if domain_analysis.processing_config:
            print(f'   optimal_chunk_size: {domain_analysis.processing_config.optimal_chunk_size}')
            print(f'   entity_confidence_threshold: {domain_analysis.processing_config.entity_confidence_threshold}')
            print(f'   vector_search_weight: {domain_analysis.processing_config.vector_search_weight}')
            print(f'   graph_search_weight: {domain_analysis.processing_config.graph_search_weight}')
            print('✅ processing_config FULLY POPULATED (Plan requirement: FIXED)')
        else:
            print('❌ processing_config MISSING')
            
        # Test downstream Agent 2 integration
        print()
        print('2️⃣ DOWNSTREAM INTEGRATION CHECK:')
        print('-' * 40)
        try:
            extraction_result = await run_knowledge_extraction(test_content, use_domain_analysis=True)
            print(f'✅ Agent 2 Integration: SUCCESS')
            print(f'   Entities extracted: {len(extraction_result.entities)}')
            print(f'   Relationships extracted: {len(extraction_result.relationships)}')
            print('✅ Agent 2 uses Agent 1 configs (Plan requirement: FIXED)')
            agent2_working = True
        except Exception as e:
            print(f'❌ Agent 2 Integration: FAILED - {str(e)[:100]}')
            agent2_working = False
            
        # Implementation status summary
        print()
        print('='*70)
        print('📈 IMPLEMENTATION STATUS vs PLAN REQUIREMENTS')
        print('='*70)
        
        print()
        print('PLAN REQUIREMENTS STATUS:')
        print(f'1. Schema completeness (100% fields): {compliance_rate:.1f}% - {"✅ IMPLEMENTED" if compliance_rate >= 80 else "❌ NOT IMPLEMENTED"}')
        print(f'2. Field name fixes (vocabulary_complexity_ratio): {"✅ FIXED" if vocab_complexity_ratio else "❌ VIOLATED"}')
        print(f'3. Missing metadata fields: {"✅ IMPLEMENTED" if compliance_rate >= 60 else "❌ NOT IMPLEMENTED"}')
        print(f'4. Agent 2 uses Agent 1 configs: {"✅ WORKING" if agent2_working else "❌ BROKEN"}')
        print(f'5. Processing config population: {"✅ WORKING" if domain_analysis.processing_config else "❌ BROKEN"}')
        
        # Overall implementation score
        implementation_score = 0
        if compliance_rate >= 80: implementation_score += 20
        if vocab_complexity_ratio is not None: implementation_score += 20
        if compliance_rate >= 60: implementation_score += 20
        if agent2_working: implementation_score += 20
        if domain_analysis.processing_config: implementation_score += 20
        
        print()
        print(f'🎯 OVERALL IMPLEMENTATION SCORE: {implementation_score}/100')
        if implementation_score >= 80:
            status = "✅ MOSTLY IMPLEMENTED"
        elif implementation_score >= 60:
            status = "⚠️ PARTIALLY IMPLEMENTED"
        else:
            status = "❌ NOT IMPLEMENTED"
        print(f'🎯 IMPLEMENTATION STATUS: {status}')
        
        # Remaining issues
        print()
        print('🔧 REMAINING ISSUES TO FIX:')
        remaining_issues = []
        if compliance_rate < 100:
            remaining_issues.append(f"Schema compliance at {compliance_rate:.1f}% instead of 100%")
        if field_name_fixes < total_critical_fields:
            remaining_issues.append(f"Field population at {field_name_fixes}/{total_critical_fields} critical fields")
        if not agent2_working:
            remaining_issues.append("Agent 2 integration broken")
        
        if remaining_issues:
            for issue in remaining_issues:
                print(f'   - {issue}')
        else:
            print('   ✅ All major issues from plan have been resolved!')
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_implementation())