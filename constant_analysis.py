import os
import re
from collections import defaultdict


def get_all_constants_from_file():
    """Extract all constants from constants.py"""
    constants = set()
    with open("agents/core/constants.py", "r") as f:
        content = f.read()
        # Find all constant definitions
        constant_patterns = [
            r"^\s*([A-Z][A-Z_0-9]*)\s*=",  # Standard constant definitions
        ]
        for pattern in constant_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            constants.update(matches)
    return constants


def count_constant_usage(constants):
    """Count usage frequency of each constant across all Python files"""
    usage_counts = defaultdict(int)

    # Get all Python files
    py_files = []
    for root, dirs, files in os.walk("."):
        # Skip virtual environments and other non-source directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in ["__pycache__", ".venv", "venv"]
        ]
        for file in files:
            if file.endswith(".py") and not file.startswith("."):
                py_files.append(os.path.join(root, file))

    # Count usage in each file
    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                for constant in constants:
                    # Count occurrences (excluding the definition line)
                    pattern = rf"\b{re.escape(constant)}\b"
                    matches = re.findall(pattern, content)
                    # Subtract 1 if this is the constants.py file (to exclude definition)
                    count = len(matches)
                    if py_file.endswith("constants.py") and count > 0:
                        count -= 1  # Exclude the definition itself
                    usage_counts[constant] += count
        except Exception as e:
            continue

    return usage_counts


# Get constants and analyze usage
print("🔍 Analyzing constant usage frequency...")
constants = get_all_constants_from_file()
usage_counts = count_constant_usage(constants)

# Sort by frequency (least used first)
sorted_constants = sorted(usage_counts.items(), key=lambda x: x[1])

print(f"\n📊 CONSTANT USAGE FREQUENCY ANALYSIS")
print(f"{'='*60}")
print(f"Total constants analyzed: {len(constants)}")
print(f"{'='*60}")

print(f"\n🔴 UNUSED CONSTANTS (0 usages):")
unused = [k for k, v in sorted_constants if v == 0]
for constant in unused:
    print(f"  • {constant}")

print(f"\n🟡 RARELY USED CONSTANTS (1-2 usages):")
rarely_used = [k for k, v in sorted_constants if 1 <= v <= 2]
for constant, count in sorted_constants:
    if 1 <= count <= 2:
        print(f"  • {constant}: {count} usage(s)")

print(f"\n🟢 MODERATELY USED CONSTANTS (3-5 usages):")
moderate = [k for k, v in sorted_constants if 3 <= v <= 5]
for constant, count in sorted_constants:
    if 3 <= count <= 5:
        print(f"  • {constant}: {count} usage(s)")

print(f"\n🔵 FREQUENTLY USED CONSTANTS (6+ usages):")
frequent = [k for k, v in sorted_constants if v >= 6]
for constant, count in sorted_constants:
    if count >= 6:
        print(f"  • {constant}: {count} usage(s)")

print(f"\n📈 SUMMARY STATISTICS:")
print(f"  • Unused (0): {len(unused)} constants")
print(f"  • Rarely used (1-2): {len(rarely_used)} constants")
print(f"  • Moderate (3-5): {len(moderate)} constants")
print(f"  • Frequent (6+): {len(frequent)} constants")
