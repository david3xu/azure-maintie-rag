#!/usr/bin/env python3
"""
Architecture compliance validator for consolidated services.
Checks import patterns and layer boundary violations.
"""

import os
import re
from pathlib import Path

def check_layer_imports(layer_path, forbidden_patterns, layer_name):
    violations = []
    
    for py_file in Path(layer_path).rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern in forbidden_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    violations.append({
                        'file': str(py_file),
                        'pattern': pattern,
                        'matches': matches
                    })
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    return violations

def main():
    # Define layer rules
    rules = {
        "API Layer": {
            "path": "api/",
            "forbidden": [
                r"from infra\.",
                r"from agents\."
            ]
        },
        "Services Layer": {
            "path": "services/",
            "forbidden": [
                r"from api\."
            ]
        },
        "Agents Layer": {
            "path": "agents/",
            "forbidden": [
                r"from services\.",
                r"from api\."
            ]
        },
        "Infrastructure Layer": {
            "path": "infra/",
            "forbidden": [
                r"from services\.",
                r"from agents\.",
                r"from api\."
            ]
        }
    }
    
    print("🔍 Architecture Compliance Validation")
    print("=" * 50)
    
    total_violations = 0
    
    for layer_name, config in rules.items():
        violations = check_layer_imports(
            config["path"], 
            config["forbidden"], 
            layer_name
        )
        
        print(f"\n{layer_name}: ", end="")
        if violations:
            print(f"❌ {len(violations)} violations")
            total_violations += len(violations)
            for violation in violations:
                print(f"  - {violation['file']}: {violation['pattern']}")
        else:
            print("✅ Clean")
    
    print(f"\n" + "=" * 50)
    if total_violations == 0:
        print("🎉 Architecture compliance: PASSED")
        return 0
    else:
        print(f"⚠️  Architecture compliance: {total_violations} violations found")
        return 1

if __name__ == "__main__":
    exit(main())