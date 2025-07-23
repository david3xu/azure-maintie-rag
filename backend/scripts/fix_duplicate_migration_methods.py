#!/usr/bin/env python3
"""
Remove duplicate _migrate_to_storage method from Azure Services Manager
Based on real codebase analysis of duplicate implementations
"""

from pathlib import Path
import re

def remove_duplicate_migration_method():
    """Remove second _migrate_to_storage implementation"""

    possible_paths = [
        Path("../integrations/azure_services.py"),
        Path("backend/integrations/azure_services.py")
    ]
    azure_services_path = None
    for path in possible_paths:
        if path.exists():
            azure_services_path = path
            break
    if azure_services_path is None:
        print("❌ Could not find integrations/azure_services.py")
        return False

    with open(azure_services_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find all lines containing method definition
    method_lines = []
    for i, line in enumerate(lines):
        if "def _migrate_to_storage(" in line:
            method_lines.append(i)

    print(f"Found _migrate_to_storage methods at lines: {[i+1 for i in method_lines]}")

    if len(method_lines) <= 1:
        print("Only one implementation found - no duplicates to remove")
        return True

    # Remove the second implementation
    second_method_start = method_lines[1]

    # Find end of second method (next method definition or class end)
    method_end = len(lines)
    for i in range(second_method_start + 1, len(lines)):
        line = lines[i]
        # Look for next method or class definition at same or lesser indentation
        if (line.strip() and
            not line.startswith('    ') and
            not line.startswith('\t') and
            not line.startswith('#')):
            method_end = i
            break
        # Or next method definition
        if line.strip().startswith('def ') and not line.startswith('        '):
            method_end = i
            break

    print(f"Removing lines {second_method_start + 1} to {method_end}")

    # Remove the duplicate method
    new_lines = lines[:second_method_start] + lines[method_end:]

    # Write back to file
    with open(azure_services_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("✅ Duplicate _migrate_to_storage method removed")
    return True

if __name__ == "__main__":
    remove_duplicate_migration_method()