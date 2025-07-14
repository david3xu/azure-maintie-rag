#!/usr/bin/env python3
"""
Debug script to test QueryType enum and query analyzer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.maintenance_models import QueryType, QueryAnalysis
from src.enhancement.query_analyzer import MaintenanceQueryAnalyzer

def test_query_type():
    """Test QueryType enum"""
    print("Testing QueryType enum...")

    # Test enum values
    print(f"QueryType.TROUBLESHOOTING = {QueryType.TROUBLESHOOTING}")
    print(f"QueryType.TROUBLESHOOTING.value = {QueryType.TROUBLESHOOTING.value}")
    print(f"type(QueryType.TROUBLESHOOTING) = {type(QueryType.TROUBLESHOOTING)}")

    # Test creating from string
    qt1 = QueryType("troubleshooting")
    print(f"QueryType('troubleshooting') = {qt1}")
    print(f"QueryType('troubleshooting').value = {qt1.value}")

    # Test creating from uppercase string
    try:
        qt2 = QueryType("TROUBLESHOOTING")
        print(f"QueryType('TROUBLESHOOTING') = {qt2}")
    except ValueError as e:
        print(f"QueryType('TROUBLESHOOTING') failed: {e}")

def test_query_analyzer():
    """Test query analyzer"""
    print("\nTesting query analyzer...")

    analyzer = MaintenanceQueryAnalyzer()

    # Test query classification
    test_queries = [
        "How to fix pump failure?",
        "What is the maintenance procedure for pump P-101?",
        "How to troubleshoot motor issues?",
        "What is the preventive maintenance schedule?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            analysis = analyzer.analyze_query(query)
            print(f"Query type: {analysis.query_type}")
            print(f"Query type value: {analysis.query_type.value}")
            print(f"Query type type: {type(analysis.query_type)}")

            # Test accessing .value attribute
            try:
                value = analysis.query_type.value
                print(f"Successfully accessed .value: {value}")
            except AttributeError as e:
                print(f"Failed to access .value: {e}")

        except Exception as e:
            print(f"Error analyzing query: {e}")

if __name__ == "__main__":
    test_query_type()
    test_query_analyzer()