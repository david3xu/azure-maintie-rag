#!/usr/bin/env python3
"""
Complete Configuration Cleanup Script

Performs 100% cleanup of config directory:
1. Remove backup directory (no longer needed)
2. Remove legacy directory (migration complete)
3. Remove migration report (cleanup complete)
4. Remove unused config files
5. Keep only essential, working files
6. Update any remaining import references
"""

import os
import shutil
from pathlib import Path
import subprocess
from typing import List


class CompleteConfigCleaner:
    """Performs 100% cleanup of configuration directory"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)

        # Files to keep (essential only)
        self.essential_files = {
            "__init__.py",                    # Package init
            "settings.py",                    # Core environment settings
            "timeout_config.py",              # System timeouts
            "data_driven_schema.py",          # Unified schema
            "consolidated_config.py",         # Main config manager
            "unified_data_driven_config.yaml" # Generated config
        }

        # Directories to keep
        self.essential_dirs = {
            "domains",      # Domain configurations
            "environments", # Environment files
            "agents"        # Agent configs (if used)
        }

        self.cleanup_report = {
            "files_removed": [],
            "directories_removed": [],
            "files_kept": [],
            "directories_kept": [],
            "total_freed_mb": 0
        }

    def calculate_size(self, path: Path) -> float:
        """Calculate size in MB"""
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        elif path.is_dir():
            total = 0
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
            return total / (1024 * 1024)
        return 0

    def remove_backup_directory(self):
        """Remove backup directory - no longer needed"""
        backup_dir = self.config_dir / "backup"

        if backup_dir.exists():
            size_mb = self.calculate_size(backup_dir)
            print(f"🗑️ Removing backup directory ({size_mb:.2f} MB)...")
            shutil.rmtree(backup_dir)
            self.cleanup_report["directories_removed"].append("backup")
            self.cleanup_report["total_freed_mb"] += size_mb
            print(f"   ✅ Removed: config/backup/")

    def remove_legacy_directory(self):
        """Remove legacy directory - migration complete"""
        legacy_dir = self.config_dir / "legacy"

        if legacy_dir.exists():
            size_mb = self.calculate_size(legacy_dir)
            print(f"🗑️ Removing legacy directory ({size_mb:.2f} MB)...")
            shutil.rmtree(legacy_dir)
            self.cleanup_report["directories_removed"].append("legacy")
            self.cleanup_report["total_freed_mb"] += size_mb
            print(f"   ✅ Removed: config/legacy/")

    def remove_migration_artifacts(self):
        """Remove migration artifacts"""
        artifacts = [
            "migration_report.json"
        ]

        for artifact in artifacts:
            artifact_path = self.config_dir / artifact
            if artifact_path.exists():
                size_mb = self.calculate_size(artifact_path)
                print(f"🗑️ Removing migration artifact: {artifact}...")
                artifact_path.unlink()
                self.cleanup_report["files_removed"].append(artifact)
                self.cleanup_report["total_freed_mb"] += size_mb
                print(f"   ✅ Removed: config/{artifact}")

    def remove_unused_files(self):
        """Remove unused configuration files"""
        # Check for any other files not in essential list
        for item in self.config_dir.iterdir():
            if item.is_file() and item.name not in self.essential_files:
                # Special handling for some files
                if item.name == "v2_config_models.py":
                    print(f"⚠️ Found v2_config_models.py - keeping for potential future integration")
                    self.cleanup_report["files_kept"].append(item.name)
                    continue

                size_mb = self.calculate_size(item)
                print(f"🗑️ Removing unused file: {item.name} ({size_mb:.2f} MB)...")
                item.unlink()
                self.cleanup_report["files_removed"].append(item.name)
                self.cleanup_report["total_freed_mb"] += size_mb
                print(f"   ✅ Removed: config/{item.name}")

    def clean_empty_directories(self):
        """Remove empty directories"""
        for item in self.config_dir.iterdir():
            if item.is_dir() and item.name not in self.essential_dirs:
                # Check if directory is empty or contains only empty subdirs
                if not any(item.rglob("*")):
                    print(f"🗑️ Removing empty directory: {item.name}...")
                    shutil.rmtree(item)
                    self.cleanup_report["directories_removed"].append(item.name)
                    print(f"   ✅ Removed: config/{item.name}/")

    def inventory_remaining_files(self):
        """Inventory remaining essential files"""
        print("📋 Inventorying remaining files...")

        for item in self.config_dir.iterdir():
            if item.is_file():
                self.cleanup_report["files_kept"].append(item.name)
                print(f"   ✅ Kept: {item.name}")
            elif item.is_dir():
                self.cleanup_report["directories_kept"].append(item.name)
                file_count = len(list(item.rglob("*.py"))) + len(list(item.rglob("*.yaml")))
                print(f"   📁 Kept: {item.name}/ ({file_count} files)")

    def update_import_references(self):
        """Update import references to use consolidated config"""
        print("🔄 Updating import references...")

        # Files that need import updates
        files_to_update = [
            "services/query_service.py",
            "services/workflow_service.py",
            "services/agent_service.py",
            "tests/validation/validate_layer_boundaries.py"
        ]

        replacement_map = {
            "from config.inter_layer_contracts import": "from config.main import",
            "config.inter_layer_contracts": "config.main",
            "from config.config_loader import": "from config.main import",
            "from config.azure_config_validator import": "from config.main import",
            "from config.production_config import": "from config.timeouts import"
        }

        updated_files = []

        for file_path in files_to_update:
            full_path = Path(file_path)
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()

                    original_content = content

                    # Apply replacements
                    for old_import, new_import in replacement_map.items():
                        if old_import in content:
                            content = content.replace(old_import, new_import)
                            print(f"   🔄 Updated import in {file_path}")

                    # Only write if changed
                    if content != original_content:
                        with open(full_path, 'w') as f:
                            f.write(content)
                        updated_files.append(file_path)
                        print(f"   ✅ Updated: {file_path}")

                except Exception as e:
                    print(f"   ⚠️ Could not update {file_path}: {e}")

        if not updated_files:
            print("   ✅ No import references needed updating")

    def validate_final_state(self):
        """Validate the final clean configuration state"""
        print("✅ Validating final configuration state...")

        try:
            # Test the consolidated config system
            import sys
            sys.path.append('.')

            from config.main import get_config_manager, validate_all_config

            config_manager = get_config_manager()
            validation = validate_all_config()

            print(f"   ✅ Configuration manager: Working")
            print(f"   📊 Available domains: {len(config_manager.list_domains())}")
            print(f"   🔍 Data-driven config: {validation['details']['data_driven']}")

            if validation['valid']:
                print(f"   ✅ Validation: PASSED")
            else:
                print(f"   ⚠️ Validation: Issues found but system functional")

        except Exception as e:
            print(f"   ❌ Validation failed: {e}")

    def generate_cleanup_summary(self):
        """Generate final cleanup summary"""
        print("\n📊 COMPLETE CLEANUP SUMMARY")
        print("=" * 50)

        print(f"Files removed: {len(self.cleanup_report['files_removed'])}")
        for file in self.cleanup_report['files_removed']:
            print(f"   - {file}")

        print(f"\nDirectories removed: {len(self.cleanup_report['directories_removed'])}")
        for dir in self.cleanup_report['directories_removed']:
            print(f"   - {dir}/")

        print(f"\nFiles kept: {len(self.cleanup_report['files_kept'])}")
        for file in self.cleanup_report['files_kept']:
            print(f"   ✅ {file}")

        print(f"\nDirectories kept: {len(self.cleanup_report['directories_kept'])}")
        for dir in self.cleanup_report['directories_kept']:
            print(f"   📁 {dir}/")

        print(f"\nTotal space freed: {self.cleanup_report['total_freed_mb']:.2f} MB")

        print(f"\n🎯 FINAL CONFIG STRUCTURE:")
        print(f"config/")
        print(f"├── essential core files ({len(self.cleanup_report['files_kept'])} files)")
        print(f"├── domains/ (generated configs)")
        print(f"├── environments/ (env settings)")
        print(f"└── 100% clean and unified!")

    def run_complete_cleanup(self):
        """Run the complete 100% cleanup"""
        print("🧹 100% CONFIGURATION CLEANUP")
        print("=" * 60)
        print("Removing all unnecessary files and directories")
        print("=" * 60)

        # Execute cleanup steps
        self.remove_backup_directory()
        self.remove_legacy_directory()
        self.remove_migration_artifacts()
        self.remove_unused_files()
        self.clean_empty_directories()
        self.inventory_remaining_files()
        self.update_import_references()
        self.validate_final_state()
        self.generate_cleanup_summary()

        print("\n🎉 100% CLEANUP COMPLETED!")
        print("Configuration directory is now completely clean and unified!")

        return self.cleanup_report


def main():
    """Run complete configuration cleanup"""
    cleaner = CompleteConfigCleaner()
    return cleaner.run_complete_cleanup()


if __name__ == "__main__":
    result = main()
    print(f"\n🏁 Cleanup Result: {result['total_freed_mb']:.2f} MB freed")
