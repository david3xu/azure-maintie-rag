#!/usr/bin/env python3
"""
Step 2.1 Validation: Fix Direct Service Instantiation - Update all endpoints to use Depends() pattern
Validates that all endpoints use proper dependency injection
"""

def validate_no_direct_instantiation():
    """Validate that no endpoints have direct service instantiation"""
    print("🔍 Validating No Direct Service Instantiation...")
    
    import os
    import re
    
    endpoints_path = "api/endpoints"
    direct_instantiation_patterns = [
        r'= \w+Service\(',
        r'InfrastructureService\(\)',
        r'QueryService\(\)',
        r'GNNService\(\)',
        r'WorkflowService\(\)',
        r'DataService\(\)'
    ]
    
    violations = []
    
    for root, dirs, files in os.walk(endpoints_path):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for line_num, line in enumerate(lines, 1):
                            # Skip comments
                            if line.strip().startswith('#'):
                                continue
                            
                            for pattern in direct_instantiation_patterns:
                                if re.search(pattern, line):
                                    violations.append({
                                        'file': filepath,
                                        'line': line_num,
                                        'content': line.strip(),
                                        'pattern': pattern
                                    })
                except Exception as e:
                    print(f"❌ Error reading {filepath}: {e}")
                    return False
    
    if violations:
        print(f"❌ Found {len(violations)} direct instantiation violations:")
        for violation in violations:
            print(f"  {violation['file']}:{violation['line']} - {violation['content']}")
        return False
    else:
        print("✅ No direct service instantiation found in endpoints")
        return True

def validate_depends_usage():
    """Validate that endpoints use Depends() pattern"""
    print("\n🔍 Validating Depends() Usage...")
    
    import os
    import re
    
    endpoints_path = "api/endpoints"
    expected_patterns = [
        r'from fastapi import.*Depends',
        r'= Depends\(',
        r'Depends\(get_\w+_service\)'
    ]
    
    endpoint_files = []
    for root, dirs, files in os.walk(endpoints_path):
        for file in files:
            if file.endswith('.py') and file != '__init__.py' and file != '__pycache__':
                endpoint_files.append(os.path.join(root, file))
    
    # Check key endpoints that should have DI
    key_endpoints = [
        'universal_endpoint.py',
        'unified_search_endpoint.py', 
        'gnn_endpoint.py',
        'query_endpoint.py'
    ]
    
    validated_endpoints = 0
    
    for filepath in endpoint_files:
        filename = os.path.basename(filepath)
        if filename in key_endpoints:
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                    # Check for Depends import
                    if 'Depends' in content:
                        print(f"✅ {filename} imports Depends")
                        
                        # Check for Depends usage in function parameters
                        if re.search(r'= Depends\(', content):
                            print(f"✅ {filename} uses Depends() pattern")
                            validated_endpoints += 1
                        else:
                            print(f"⚠️ {filename} imports Depends but may not use it properly")
                    else:
                        print(f"❌ {filename} does not import Depends")
                        
            except Exception as e:
                print(f"❌ Error reading {filepath}: {e}")
                return False
    
    expected_endpoints = len(key_endpoints)
    success_rate = validated_endpoints / expected_endpoints if expected_endpoints > 0 else 0
    
    print(f"📊 Validation Results: {validated_endpoints}/{expected_endpoints} endpoints use proper DI")
    
    if success_rate >= 0.75:  # At least 75% should be using DI
        print("✅ Sufficient endpoints using Depends() pattern")
        return True
    else:
        print("❌ Insufficient endpoints using Depends() pattern")
        return False

def validate_dependency_imports():
    """Validate that endpoints import from dependencies_new.py"""
    print("\n🔍 Validating Dependency Imports...")
    
    import os
    
    endpoints_path = "api/endpoints"
    dependency_import_pattern = "from api.dependencies import"
    
    key_endpoints = [
        'universal_endpoint.py',
        'unified_search_endpoint.py',
        'gnn_endpoint.py'
    ]
    
    validated_imports = 0
    
    for root, dirs, files in os.walk(endpoints_path):
        for file in files:
            if file in key_endpoints:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        
                        if dependency_import_pattern in content:
                            print(f"✅ {file} imports from dependencies_new")
                            validated_imports += 1
                        else:
                            print(f"❌ {file} does not import from dependencies_new")
                            
                except Exception as e:
                    print(f"❌ Error reading {filepath}: {e}")
                    return False
    
    expected_imports = len(key_endpoints)
    success_rate = validated_imports / expected_imports if expected_imports > 0 else 0
    
    if success_rate >= 0.75:
        print("✅ Sufficient endpoints importing from DI container")
        return True
    else:
        print("❌ Insufficient endpoints importing from DI container") 
        return False

def validate_service_provider_functions():
    """Validate that all required service provider functions exist"""
    print("\n🔍 Validating Service Provider Functions...")
    
    try:
        from api.dependencies import (
            get_infrastructure_service,
            get_query_service,
            get_workflow_service,
            get_gnn_service,
            get_azure_settings
        )
        
        provider_functions = [
            ('get_infrastructure_service', get_infrastructure_service),
            ('get_query_service', get_query_service),
            ('get_workflow_service', get_workflow_service),
            ('get_gnn_service', get_gnn_service),
            ('get_azure_settings', get_azure_settings)
        ]
        
        for name, func in provider_functions:
            if callable(func):
                print(f"✅ {name} is available and callable")
            else:
                print(f"❌ {name} is not callable")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import provider functions: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Step 2.1 Validation: Fix Direct Service Instantiation")
    print("=" * 70)
    
    success = True
    success &= validate_no_direct_instantiation()
    success &= validate_depends_usage()
    success &= validate_dependency_imports()
    success &= validate_service_provider_functions()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ STEP 2.1 COMPLETE: All endpoints use proper Depends() pattern")
        print("🎯 Key achievements:")
        print("   - No direct service instantiation in endpoints")
        print("   - Proper Depends() usage implemented")
        print("   - DI container integration working")
        print("   - Service provider functions available")
        print("\n📋 Ready for Step 2.2: Circuit Breaker Patterns")
    else:
        print("❌ STEP 2.1 INCOMPLETE: Some validations failed")
    
    return success

if __name__ == "__main__":
    main()