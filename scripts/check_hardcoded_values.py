#!/usr/bin/env python3
"""
Pre-commit hook to detect hardcoded values in agents/ directory.

This script enforces the zero-hardcoded-values philosophy by detecting:
- Magic numbers (except 0, 1, -1, and common array indices)
- Hardcoded thresholds, timeouts, limits
- Configuration values that should be centralized
- Business logic parameters

Exemptions:
- Constants defined in agents/core/constants.py
- Obvious array indices (0, 1, 2)
- Boolean values and None
- Test files (_test.py, test_.py)
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple


class HardcodedValueDetector(ast.NodeVisitor):
    """AST visitor to detect hardcoded values in Python code."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.violations: List[Tuple[int, str, str]] = []
        # Allow common array indices, HTTP status codes, and basic values
        self.allowed_numbers = {0, 1, -1, 2, 3, 4, 5, 10, 100, 200, 201, 400, 401, 403, 404, 500}
        self.allowed_strings = {
            "utf-8", "ascii", "json", "text", "html", "xml", 
            "get", "post", "put", "delete", "patch", "head", "options",
            "error", "warning", "info", "debug", "critical",
            "true", "false", "none", "null", "ok", "success", "failed",
            "", " ", "\n", "\t", "\\n", "\\t"
        }
        # Context to detect if we're in a suspicious location for business logic
        self._in_business_logic_context = False
        
    def visit_Num(self, node):
        """Visit numeric literals."""
        if hasattr(node, 'n') and isinstance(node.n, (int, float)):
            if self._is_suspicious_number(node.n):
                self._add_violation(node.lineno, f"Magic number: {node.n}", 
                                  "Move to agents/core/constants.py")
        self.generic_visit(node)
    
    def visit_Constant(self, node):
        """Visit constant values (Python 3.8+)."""
        if isinstance(node.value, (int, float)):
            if self._is_suspicious_number(node.value):
                self._add_violation(node.lineno, f"Magic number: {node.value}", 
                                  "Move to agents/core/constants.py")
        elif isinstance(node.value, str):
            if self._is_suspicious_string(node.value):
                self._add_violation(node.lineno, f"Hardcoded string: '{node.value}'", 
                                  "Move to configuration or constants")
        self.generic_visit(node)
    
    def visit_Str(self, node):
        """Visit string literals (Python < 3.8)."""
        if hasattr(node, 's') and self._is_suspicious_string(node.s):
            self._add_violation(node.lineno, f"Hardcoded string: '{node.s}'", 
                              "Move to configuration or constants")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function calls to detect problematic patterns."""
        # Check for asyncio.sleep with hardcoded values
        if (isinstance(node.func, ast.Attribute) and 
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'asyncio' and 
            node.func.attr == 'sleep' and
            node.args and isinstance(node.args[0], (ast.Num, ast.Constant))):
            
            value = getattr(node.args[0], 'n', getattr(node.args[0], 'value', None))
            if value and value not in self.allowed_numbers:
                self._add_violation(node.lineno, f"Hardcoded sleep duration: {value}", 
                                  "Use processing delay constants")
        
        # Check for time.sleep patterns
        if (isinstance(node.func, ast.Attribute) and 
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'time' and 
            node.func.attr == 'sleep'):
            
            if node.args and isinstance(node.args[0], (ast.Num, ast.Constant)):
                value = getattr(node.args[0], 'n', getattr(node.args[0], 'value', None))
                if value and value not in self.allowed_numbers:
                    self._add_violation(node.lineno, f"Hardcoded sleep duration: {value}", 
                                      "Use processing delay constants")
        
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        """Visit comparison operations to detect threshold comparisons."""
        # Look for threshold comparisons like: confidence > 0.7
        for comparator in node.comparators:
            if isinstance(comparator, (ast.Num, ast.Constant)):
                value = getattr(comparator, 'n', getattr(comparator, 'value', None))
                if isinstance(value, float) and 0.0 < value < 1.0:
                    self._add_violation(comparator.lineno, f"Hardcoded threshold: {value}", 
                                      "Use centralized threshold constants")
        self.generic_visit(node)
    
    def _is_suspicious_number(self, value: float) -> bool:
        """Check if a number looks like it should be configurable."""
        # Skip allowed numbers
        if value in self.allowed_numbers:
            return False
            
        # Flag decimal values that look like thresholds, rates, or percentages
        if isinstance(value, float):
            # Common threshold/confidence ranges (0.1 to 0.99)
            if 0.1 <= value <= 0.99:
                return True
            # Common timeout/delay ranges (0.01 to 60 seconds)
            if 0.01 <= value <= 60.0 and value not in {0.1, 0.5, 1.0}:
                return True
                
        # Flag integers that look like limits, counts, or sizes
        if isinstance(value, int) and value > 10:
            # Common limit ranges (20-10000)
            if 20 <= value <= 10000:
                return True
            # Large numbers that might be buffer sizes, timeouts in ms
            if value >= 1000 and value % 100 == 0:  # Round numbers like 1000, 5000, etc.
                return True
                
        return False
    
    def _is_suspicious_string(self, s: str) -> bool:
        """Check if a string looks like it should be configurable."""
        s_lower = s.lower().strip()
        
        # Skip allowed strings
        if s_lower in self.allowed_strings or len(s) <= 1:
            return False
            
        # Skip docstrings and multi-line strings (likely documentation)
        if '\n' in s or len(s) > 100:
            return False
            
        # Skip log messages and display strings (contain common words)
        if any(word in s_lower for word in [
            'initialize', 'initializing', 'failed', 'error', 'success', 
            'complete', 'loading', 'processing', 'found', 'created',
            'executing', 'finished', 'starting', 'stopping'
        ]):
            return False
            
        # Only flag URLs that look like they should be configurable
        if re.match(r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s) and not any(domain in s_lower for domain in [
            'example.com', 'localhost', 'test.com', 'api.example'
        ]):
            return True
            
        # Only flag Azure resource names that look like hardcoded endpoints
        if re.match(r'[a-zA-Z0-9-]+\.openai\.azure\.com', s):
            return True
        if re.match(r'[a-zA-Z0-9-]+\.search\.windows\.net', s):
            return True
        if re.match(r'[a-zA-Z0-9-]+\.documents\.azure\.com', s):
            return True
            
        # Detect obvious configuration values (but not variable names)
        if (len(s) < 50 and 
            any(pattern in s_lower for pattern in [
                'api-version', 'api_version', 'content-type',
                '.json', '.yaml', '.yml'
            ]) and
            not s_lower.replace('_', '').replace('-', '').isalpha()):
            return True
            
        return False
    
    def _add_violation(self, lineno: int, description: str, suggestion: str):
        """Add a violation to the list."""
        self.violations.append((lineno, description, suggestion))


def check_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Check a single Python file for hardcoded values."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip empty files
        if not content.strip():
            return []
            
        # Parse the AST
        tree = ast.parse(content, filename=str(filepath))
        
        # Run the detector
        detector = HardcodedValueDetector(str(filepath))
        detector.visit(tree)
        
        return detector.violations
        
    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}", "Fix syntax error")]
    except Exception as e:
        return [(0, f"Error processing file: {e}", "Check file manually")]


def is_exempted_file(filepath: Path) -> bool:
    """Check if a file should be exempted from hardcoded value checks."""
    
    # Exempt constants.py - this is where constants should be defined
    if filepath.name == 'constants.py':
        return True
        
    # Exempt test files
    if 'test' in filepath.name.lower() or filepath.name.startswith('test_'):
        return True
        
    # Exempt __init__.py files (usually just imports)
    if filepath.name == '__init__.py':
        return True
        
    # Exempt analysis/documentation files
    if filepath.suffix.lower() in {'.md', '.rst', '.txt'}:
        return True
        
    return False


def main():
    """Main function to check all Python files in agents/ directory."""
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    agents_dir = project_root / 'agents'
    
    if not agents_dir.exists():
        print("❌ Error: agents/ directory not found")
        return 1
    
    # Find all Python files in agents/
    python_files = list(agents_dir.rglob('*.py'))
    
    if not python_files:
        print("✅ No Python files found in agents/ directory")
        return 0
    
    total_violations = 0
    files_with_violations = 0
    
    print("🔍 Checking agents/ directory for hardcoded values...")
    print()
    
    for filepath in sorted(python_files):
        # Skip exempted files
        if is_exempted_file(filepath):
            continue
            
        violations = check_file(filepath)
        
        if violations:
            files_with_violations += 1
            total_violations += len(violations)
            
            # Show relative path from project root
            rel_path = filepath.relative_to(project_root)
            print(f"❌ {rel_path}")
            
            for lineno, description, suggestion in violations:
                print(f"   Line {lineno}: {description}")
                print(f"   💡 {suggestion}")
                print()
    
    # Summary
    if total_violations == 0:
        print("✅ No hardcoded values detected in agents/ directory!")
        print("🎉 Zero-hardcoded-values philosophy maintained!")
        return 0
    else:
        print(f"❌ Found {total_violations} hardcoded values in {files_with_violations} files")
        print()
        print("🛠️  Fix suggestions:")
        print("   • Move magic numbers to agents/core/constants.py")
        print("   • Use centralized configuration classes")
        print("   • Load thresholds from dynamic_config_manager")
        print("   • Use processing delay constants for sleep operations")
        print()
        print("📚 See agents/core/constants.py for examples of proper centralization")
        return 1


if __name__ == '__main__':
    sys.exit(main())