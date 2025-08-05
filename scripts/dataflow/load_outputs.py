#!/usr/bin/env python3
"""
Simple Output Loader - CODING_STANDARDS Compliant
Clean output validation script without over-engineering.
"""

import json
import sys
from pathlib import Path


def load_pipeline_outputs():
    """Simple pipeline output loader"""
    print("📊 Loading Pipeline Outputs")
    
    try:
        # Check for common output directories
        data_dir = Path("data")
        outputs_dir = data_dir / "outputs"
        
        if not outputs_dir.exists():
            print("❌ No outputs directory found")
            return None
            
        # Find JSON result files
        json_files = list(outputs_dir.glob("**/*.json"))
        
        if not json_files:
            print("❌ No JSON output files found")
            return None
            
        print(f"📂 Found {len(json_files)} output files")
        
        # Load and validate files
        results = {}
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                results[json_file.name] = {
                    "path": str(json_file),
                    "size": json_file.stat().st_size,
                    "keys": list(data.keys()) if isinstance(data, dict) else "non-dict",
                    "valid": True
                }
                print(f"✅ Loaded: {json_file.name}")
                
            except Exception as e:
                print(f"⚠️ Failed to load {json_file.name}: {e}")
                results[json_file.name] = {"valid": False, "error": str(e)}
        
        print(f"📈 Successfully loaded {len([r for r in results.values() if r.get('valid')])} files")
        return results
        
    except Exception as e:
        print(f"❌ Output loading failed: {e}")
        return None


if __name__ == "__main__":
    result = load_pipeline_outputs()
    
    if result:
        print("\n🎯 Output Summary:")
        for filename, info in result.items():
            if info.get("valid"):
                print(f"   📄 {filename}: {info['size']} bytes")
    
    sys.exit(0 if result else 1)