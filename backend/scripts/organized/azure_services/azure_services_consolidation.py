#!/usr/bin/env python3
"""
Azure Services Manager Architecture Consolidation
Enterprise service implementation pattern cleanup
"""

import re
from pathlib import Path

def consolidate_azure_services_architecture():
    """
    Consolidate Azure Services Manager to single enterprise implementation
    Based on real codebase duplicate service pattern analysis
    """

    azure_services_path = Path("../integrations/azure_services.py").resolve()
    if not azure_services_path.exists():
        # Try with correct relative path from project root
        azure_services_path = Path("backend/integrations/azure_services.py").resolve()
        if not azure_services_path.exists():
            print(f"❌ Azure Services file not found: {azure_services_path}")
            return False

    # Read current implementation
    with open(azure_services_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Identify legacy implementation boundaries (approximate line range 902-950)
    # Look for the second _migrate_to_storage method definition

    # Pattern to find duplicate method definitions
    method_pattern = r'def _migrate_to_storage\(self, source_data_path: str, domain: str, migration_context: Dict\)'

    matches = list(re.finditer(method_pattern, content))

    if len(matches) > 1:
        print(f"🔍 Found {len(matches)} _migrate_to_storage implementations")

        # Keep only the first (modern) implementation
        # Find the start of the second implementation
        second_method_start = matches[1].start()

        # Find the next method definition or class end to determine boundaries
        # Look for next 'def ' or 'class ' or end of file
        next_definition_pattern = r'\n    def \w+\(|class \w+:|^[^\s]'

        # Search from the second method start
        next_match = re.search(next_definition_pattern, content[second_method_start + 100:])

        if next_match:
            # Calculate actual position
            second_method_end = second_method_start + 100 + next_match.start()
        else:
            # If no next definition found, assume it goes to end of class/file
            # Find the end of the current indentation level
            lines = content[second_method_start:].split('\n')
            method_end_line = 0
            for i, line in enumerate(lines[1:], 1):  # Skip first line (method definition)
                if line and not line.startswith('    '):  # End of method indentation
                    method_end_line = i
                    break

            if method_end_line == 0:
                method_end_line = len(lines)

            second_method_end = second_method_start + len('\n'.join(lines[:method_end_line]))

        # Remove the duplicate implementation
        content_before = content[:second_method_start]
        content_after = content[second_method_end:]

        # Add comment explaining the consolidation
        consolidation_comment = '''
    # Note: Legacy _migrate_to_storage implementation removed during
    # Azure Services Manager architecture consolidation
    # Enterprise pattern: Single service implementation per interface
'''

        consolidated_content = content_before + consolidation_comment + content_after

        # Write consolidated implementation
        with open(azure_services_path, 'w', encoding='utf-8') as f:
            f.write(consolidated_content)

        print("✅ Azure Services Manager architecture consolidated")
        print("📊 Service implementation pattern: Modern enterprise implementation retained")
        print("🗑️  Legacy implementation removed")
        return True

    else:
        print("ℹ️  Single implementation found - architecture already consolidated")
        return True

def validate_azure_services_architecture():
    """Validate Azure Services Manager service interface consistency"""

    azure_services_path = Path("../integrations/azure_services.py").resolve()
    if not azure_services_path.exists():
        azure_services_path = Path("backend/integrations/azure_services.py").resolve()
        if not azure_services_path.exists():
            print(f"❌ Azure Services file not found: {azure_services_path}")
            return False

    with open(azure_services_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for duplicate method implementations
    method_pattern = r'def _migrate_to_storage\('
    matches = re.findall(method_pattern, content)

    print(f"🔍 Azure Services Architecture Validation:")
    print(f"  Migration service implementations: {len(matches)}")

    if len(matches) == 1:
        print("✅ Enterprise service pattern: Single implementation per interface")
        return True
    else:
        print(f"⚠️  Multiple implementations detected: {len(matches)}")
        return False

if __name__ == "__main__":
    print("🏗️ Azure Services Manager Architecture Consolidation")
    print("=" * 60)

    # Validate current architecture
    if not validate_azure_services_architecture():
        print("\n🔧 Applying enterprise architecture consolidation...")
        success = consolidate_azure_services_architecture()
        print("\n📊 Post-consolidation validation:")
        validate_azure_services_architecture()

        if success:
            print("\n✅ Azure Services Manager architecture successfully consolidated")
            exit(0)
        else:
            print("\n❌ Architecture consolidation failed")
            exit(1)
    else:
        print("\n✅ Azure Services Manager architecture is already consolidated")
        exit(0)