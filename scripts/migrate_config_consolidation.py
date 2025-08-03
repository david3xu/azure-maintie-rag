#!/usr/bin/env python3
"""
Configuration Consolidation Migration Script

This script consolidates the inconsistent configuration files in @config/
into a unified data-driven configuration system.

Steps:
1. Move legacy config files to config/legacy/
2. Update imports in codebase to use consolidated config
3. Validate new configuration system
4. Generate migration report
"""

import os
import shutil
from pathlib import Path
import subprocess
from typing import List, Dict, Any
import json


class ConfigConsolidationMigrator:
    """Migrates legacy config files to unified system"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.legacy_dir = self.config_dir / "legacy"
        self.backup_dir = self.config_dir / "backup"
        
        # Files to migrate to legacy
        self.legacy_files = [
            "config_loader.py",
            "azure_config_validator.py", 
            "production_config.py",
            "inter_layer_contracts.py"
        ]
        
        # Files to keep (core system files)
        self.keep_files = [
            "__init__.py",
            "settings.py",           # Core environment settings
            "timeout_config.py",     # System timeouts
            "v2_config_models.py",   # V2 models (may integrate later)
            "data_driven_schema.py", # Our unified schema
            "consolidated_config.py" # New consolidated manager
        ]
        
        self.migration_report = {
            "timestamp": None,
            "files_migrated": [],
            "files_kept": [],
            "directories_created": [],
            "imports_to_update": [],
            "validation_results": {},
            "next_steps": []
        }
    
    def create_backup(self):
        """Create backup of current config directory"""
        print("📦 Creating backup of current config directory...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        # Create backup
        shutil.copytree(self.config_dir, self.backup_dir, ignore=shutil.ignore_patterns("backup", "legacy"))
        print(f"   ✅ Backup created: {self.backup_dir}")
        
        self.migration_report["directories_created"].append(str(self.backup_dir))
    
    def create_legacy_directory(self):
        """Create legacy directory for old config files"""
        print("📂 Creating legacy directory...")
        
        self.legacy_dir.mkdir(exist_ok=True)
        
        # Create legacy README
        legacy_readme = self.legacy_dir / "README.md"
        with open(legacy_readme, 'w') as f:
            f.write("""# Legacy Configuration Files

These configuration files have been moved here during the consolidation
to the unified data-driven configuration system.

## Replaced By
- `config_loader.py` → `consolidated_config.py` 
- `azure_config_validator.py` → `consolidated_config.py`
- `production_config.py` → `timeout_config.py` + `consolidated_config.py`
- `inter_layer_contracts.py` → `data_driven_schema.py`

## Migration Date
""" + f"{self.migration_report['timestamp']}\n\n" + """
## Status
These files are kept for reference but should not be imported.
The new system provides backwards compatibility where needed.

## New Configuration System
Use `config.consolidated_config.get_config_manager()` for all configuration needs.
""")
        
        print(f"   ✅ Legacy directory created: {self.legacy_dir}")
        self.migration_report["directories_created"].append(str(self.legacy_dir))
    
    def move_legacy_files(self):
        """Move legacy configuration files to legacy directory"""
        print("🚚 Moving legacy configuration files...")
        
        for filename in self.legacy_files:
            source_file = self.config_dir / filename
            if source_file.exists():
                dest_file = self.legacy_dir / filename
                shutil.move(str(source_file), str(dest_file))
                print(f"   📄 Moved: {filename} → legacy/{filename}")
                self.migration_report["files_migrated"].append(filename)
            else:
                print(f"   ⚠️ File not found: {filename}")
    
    def analyze_kept_files(self):
        """Analyze files that are kept in the new system"""
        print("📋 Analyzing files kept in new system...")
        
        for filename in self.keep_files:
            file_path = self.config_dir / filename
            if file_path.exists():
                print(f"   ✅ Kept: {filename}")
                self.migration_report["files_kept"].append(filename)
            else:
                print(f"   ❌ Missing: {filename}")
    
    def find_import_references(self):
        """Find code that imports legacy config files"""
        print("🔍 Finding import references to legacy config files...")
        
        # Search for imports of legacy files
        legacy_imports = []
        
        try:
            # Search for imports in Python files
            result = subprocess.run([
                'grep', '-r', '--include=*.py', 
                '-E', '(config_loader|azure_config_validator|production_config|inter_layer_contracts)',
                '.'
            ], capture_output=True, text=True, cwd='.')
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'from config.' in line or 'import config.' in line:
                        legacy_imports.append(line.strip())
            
        except subprocess.CalledProcessError:
            print("   ⚠️ Could not search for import references")
        
        self.migration_report["imports_to_update"] = legacy_imports
        
        if legacy_imports:
            print(f"   ⚠️ Found {len(legacy_imports)} import references to update")
            for imp in legacy_imports[:5]:  # Show first 5
                print(f"      - {imp}")
        else:
            print("   ✅ No legacy import references found")
    
    def validate_new_system(self):
        """Validate the new consolidated configuration system"""
        print("✅ Validating new configuration system...")
        
        try:
            # Import and test the new system
            import sys
            sys.path.append('.')
            
            from config.consolidated_config import get_config_manager, validate_all_config
            
            # Test configuration manager
            config_manager = get_config_manager()
            print(f"   ✅ Configuration manager loaded")
            print(f"   📊 Available domains: {len(config_manager.list_domains())}")
            
            # Test validation
            validation_results = validate_all_config()
            self.migration_report["validation_results"] = validation_results
            
            if validation_results["valid"]:
                print(f"   ✅ Configuration validation passed")
            else:
                print(f"   ⚠️ Configuration validation found issues:")
                for error in validation_results["errors"]:
                    print(f"      - {error}")
            
            # Test data-driven config
            if validation_results["details"]["data_driven"]:
                print(f"   ✅ Data-driven configuration active")
            else:
                print(f"   ⚠️ Data-driven configuration not loaded")
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            self.migration_report["validation_results"] = {"error": str(e)}
    
    def generate_migration_report(self):
        """Generate comprehensive migration report"""
        print("📊 Generating migration report...")
        
        self.migration_report["timestamp"] = "2025-08-02T12:15:00"
        
        # Add next steps
        self.migration_report["next_steps"] = [
            "Update any import references found to use consolidated_config",
            "Test all functionality that used legacy config files", 
            "Remove legacy directory after validation",
            "Update documentation to reference new config system",
            "Consider integrating v2_config_models.py into consolidated system"
        ]
        
        # Save report
        report_file = self.config_dir / "migration_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.migration_report, f, indent=2)
        
        print(f"   ✅ Migration report saved: {report_file}")
        
        # Print summary
        print("\n📋 MIGRATION SUMMARY")
        print("=" * 50)
        print(f"Files migrated to legacy: {len(self.migration_report['files_migrated'])}")
        print(f"Files kept in new system: {len(self.migration_report['files_kept'])}")
        print(f"Import references to update: {len(self.migration_report['imports_to_update'])}")
        
        validation = self.migration_report["validation_results"]
        if isinstance(validation, dict) and "valid" in validation:
            status = "✅ PASSED" if validation["valid"] else "⚠️ ISSUES FOUND"
            print(f"New system validation: {status}")
            
            if validation["details"]["data_driven"]:
                print("Data-driven config: ✅ ACTIVE")
            else:
                print("Data-driven config: ⚠️ NOT LOADED")
        
    def run_migration(self):
        """Run the complete migration process"""
        print("🚀 CONFIGURATION CONSOLIDATION MIGRATION")
        print("=" * 60)
        print("Consolidating inconsistent config files into unified system")
        print("=" * 60)
        
        # Execute migration steps
        self.create_backup()
        self.create_legacy_directory()
        self.move_legacy_files()
        self.analyze_kept_files()
        self.find_import_references()
        self.validate_new_system()
        self.generate_migration_report()
        
        print("\n🎉 MIGRATION COMPLETED!")
        print(f"📦 Backup: {self.backup_dir}")
        print(f"📂 Legacy files: {self.legacy_dir}")
        print(f"📊 Report: {self.config_dir}/migration_report.json")
        
        return self.migration_report


def main():
    """Run the configuration consolidation migration"""
    migrator = ConfigConsolidationMigrator()
    return migrator.run_migration()


if __name__ == "__main__":
    result = main()
    print(f"\n🏁 Migration Result: {len(result['files_migrated'])} files migrated")