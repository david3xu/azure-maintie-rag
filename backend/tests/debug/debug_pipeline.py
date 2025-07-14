#!/usr/bin/env python3
"""
Debug script to trace the pipeline and find where QueryType enum becomes a string
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.maintenance_models import QueryType, QueryAnalysis, EnhancedQuery
from src.enhancement.query_analyzer import MaintenanceQueryAnalyzer
from src.generation.llm_interface import MaintenanceLLMInterface

def test_pipeline():
    """Test the full pipeline to find where the issue occurs"""
    print("Testing full pipeline...")

    # Initialize components
    analyzer = MaintenanceQueryAnalyzer()
    llm_interface = MaintenanceLLMInterface()

    # Test query
    query = "How to fix pump failure?"
    print(f"\nQuery: {query}")

    try:
        # Step 1: Analyze query
        print("\nStep 1: Analyzing query...")
        analysis = analyzer.analyze_query(query)
        print(f"Analysis query_type: {analysis.query_type}")
        print(f"Analysis query_type type: {type(analysis.query_type)}")
        print(f"Analysis query_type.value: {analysis.query_type.value}")

        # Step 2: Enhance query
        print("\nStep 2: Enhancing query...")
        enhanced = analyzer.enhance_query(analysis)
        print(f"Enhanced analysis query_type: {enhanced.analysis.query_type}")
        print(f"Enhanced analysis query_type type: {type(enhanced.analysis.query_type)}")

        try:
            value = enhanced.analysis.query_type.value
            print(f"Enhanced analysis query_type.value: {value}")
        except AttributeError as e:
            print(f"ERROR: enhanced.analysis.query_type.value failed: {e}")
            print(f"This is where the issue occurs!")

        # Step 3: Test LLM interface
        print("\nStep 3: Testing LLM interface...")
        try:
            # Create mock search results
            from src.models.maintenance_models import SearchResult
            mock_results = [
                SearchResult(
                    doc_id="test1",
                    title="Test Document",
                    content="Test content about pump maintenance",
                    score=0.8,
                    source="vector"
                )
            ]

            # This should trigger the error
            result = llm_interface.generate_response(enhanced, mock_results)
            print("LLM interface call succeeded")

        except Exception as e:
            print(f"LLM interface call failed: {e}")
            print(f"Error type: {type(e)}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

def test_serialization():
    """Test if serialization/deserialization is causing the issue"""
    print("\nTesting serialization...")

    analyzer = MaintenanceQueryAnalyzer()
    query = "How to fix pump failure?"

    # Create analysis
    analysis = analyzer.analyze_query(query)
    print(f"Original query_type: {analysis.query_type}")
    print(f"Original query_type type: {type(analysis.query_type)}")

    # Test to_dict and from_dict
    analysis_dict = analysis.to_dict()
    print(f"Serialized query_type: {analysis_dict['query_type']}")
    print(f"Serialized query_type type: {type(analysis_dict['query_type'])}")

    # Test if this is where the issue occurs
    try:
        # Simulate what might happen in the pipeline
        if isinstance(analysis_dict['query_type'], str):
            print("WARNING: query_type was serialized to string!")
            # Try to convert back
            try:
                restored_type = QueryType(analysis_dict['query_type'])
                print(f"Restored type: {restored_type}")
                print(f"Restored type.value: {restored_type.value}")
            except Exception as e:
                print(f"Failed to restore: {e}")
    except Exception as e:
        print(f"Serialization test failed: {e}")

if __name__ == "__main__":
    test_pipeline()
    test_serialization()