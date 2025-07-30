#!/usr/bin/env python3
"""Minimal syntax test for modified core files"""

import ast
import sys

def test_file_syntax(filepath):
    """Test Python syntax of a file"""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source)
        print(f"✅ {filepath}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"❌ {filepath}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {filepath}: Error reading file: {e}")
        return False

if __name__ == "__main__":
    files_to_test = [
        "core/azure_storage/storage_client.py",
        "core/azure_search/search_client.py", 
        "core/azure_cosmos/cosmos_gremlin_client.py",
        "core/azure_ml/ml_client.py",
        "core/azure_ml/gnn/feature_engineering.py"
    ]
    
    print("Testing syntax of modified files...")
    all_passed = True
    
    for file_path in files_to_test:
        if not test_file_syntax(file_path):
            all_passed = False
    
    if all_passed:
        print("\n✅ All files passed syntax validation")
        sys.exit(0)
    else:
        print("\n❌ Some files failed syntax validation")
        sys.exit(1)