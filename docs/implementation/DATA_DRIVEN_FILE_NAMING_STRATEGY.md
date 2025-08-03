# Data-Driven File Naming Strategy

**Date**: August 3, 2025
**Purpose**: Define file naming strategy that follows "Data-Driven Everything" principle
**Requirement**: Zero hardcoded file names, all names generated from discovered data

## Problem Statement

The previous config structure contained **hardcoded file names** that violate the "Data-Driven Everything" principle:

```bash
# ❌ HARDCODED FILE NAMES (WRONG)
programming_language_thresholds.json
programming_language_sla.json
zero_hardcoded_validation.json
```

These names assume specific domains and file purposes, violating the universal design principle.

## Data-Driven File Naming Solution

### 1. Domain-Based Dynamic Naming

**Pattern**: `{discovered_domain_name}_{file_type}.{extension}`

```python
# ✅ DATA-DRIVEN NAMING (CORRECT)
class DataDrivenFileNaming:
    """Generate file names based on discovered data, not hardcoded assumptions"""

    def generate_domain_file_name(self, domain_path: str, file_type: str, extension: str = "json") -> str:
        """Generate file name from discovered domain directory"""
        domain_name = Path(domain_path).name.lower().replace('-', '_')
        return f"{domain_name}_{file_type}.{extension}"

    def generate_timestamped_file_name(self, file_type: str, extension: str = "json") -> str:
        """Generate file name with timestamp for temporal files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{file_type}.{extension}"

    def generate_analysis_file_name(self, analysis_type: str, domain_name: str, extension: str = "json") -> str:
        """Generate analysis file name from discovered analysis type and domain"""
        return f"{domain_name}_{analysis_type}_analysis.{extension}"

# Example usage with discovered domain "programming_language" from "Programming-Language/" directory:
namer = DataDrivenFileNaming()

# ✅ Generated from domain discovery
threshold_file = namer.generate_domain_file_name("data/raw/Programming-Language", "thresholds")
# Result: "programming_language_thresholds.json"

sla_file = namer.generate_domain_file_name("data/raw/Programming-Language", "sla")
# Result: "programming_language_sla.json"

validation_file = namer.generate_timestamped_file_name("zero_hardcoded_validation")
# Result: "20250803_142530_zero_hardcoded_validation.json"
```

### 2. Content-Driven File Organization

**Pattern**: Organize files based on discovered content characteristics, not predetermined categories

```python
class ContentDrivenOrganization:
    """Organize files based on discovered content patterns, not hardcoded categories"""

    def determine_file_organization(self, domain_analysis: StatisticalAnalysis) -> Dict[str, str]:
        """Determine file organization based on content analysis"""

        organization = {}

        # Determine complexity-based subdirectory
        if domain_analysis.technical_term_density > 0.3:
            complexity_dir = "high_complexity"
        elif domain_analysis.technical_term_density > 0.1:
            complexity_dir = "medium_complexity"
        else:
            complexity_dir = "low_complexity"

        # Determine content-type-based subdirectory
        if len(domain_analysis.n_gram_patterns) > 500:
            content_type_dir = "pattern_rich"
        elif domain_analysis.vocabulary_size > 10000:
            content_type_dir = "vocabulary_rich"
        else:
            content_type_dir = "standard_content"

        organization["complexity_path"] = complexity_dir
        organization["content_type_path"] = content_type_dir

        return organization

# Example usage:
organizer = ContentDrivenOrganization()
analysis = StatisticalAnalysis(technical_term_density=0.35, vocabulary_size=15000, ...)
org = organizer.determine_file_organization(analysis)

# ✅ Generated file path based on content characteristics
file_path = f"config/generated/learned_models/{org['complexity_path']}/{org['content_type_path']}/programming_language_thresholds.json"
# Result: "config/generated/learned_models/high_complexity/vocabulary_rich/programming_language_thresholds.json"
```

### 3. Enhanced Config Structure with Data-Driven Naming

**Updated Structure with Dynamic Naming:**

```
config/generated/
├── domains/
│   └── {discovered_domain}/           # ✅ From subdirectory discovery
│       ├── {domain}_complete_config.yaml
│       ├── {domain}_extraction_config.yaml
│       ├── {domain}_statistical_analysis.json
│       ├── {domain}_performance_model.json
│       └── {domain}_classification_rules.json
│
├── learned_models/
│   ├── {complexity_level}/            # ✅ From content analysis (high/medium/low_complexity)
│   │   └── {content_type}/            # ✅ From content analysis (pattern_rich/vocabulary_rich/standard)
│   │       ├── {domain}_thresholds.json
│   │       ├── {domain}_clustering.json
│   │       ├── {domain}_coherence.json
│   │       ├── {domain}_sla.json
│   │       ├── {domain}_cache.json
│   │       ├── {domain}_optimization.json
│   │       ├── {domain}_patterns.json
│   │       └── {domain}_rules.json
│
├── validation_reports/
│   ├── {timestamp}_zero_hardcoded_validation.json     # ✅ Timestamped
│   ├── {timestamp}_learning_confidence_report.json   # ✅ Timestamped
│   ├── {timestamp}_configuration_quality.json        # ✅ Timestamped
│   └── {timestamp}_domain_coverage.json              # ✅ Timestamped
│
└── unified_configs/
    └── {discovered_domain}_unified.yaml              # ✅ From domain discovery
```

### 4. Implementation in Config Manager

```python
# config/main.py - Enhanced with data-driven naming
class EnhancedConfigurationManager:

    def __init__(self):
        self.file_namer = DataDrivenFileNaming()
        self.organizer = ContentDrivenOrganization()

    async def save_learned_model(
        self,
        domain_path: str,
        model_data: Any,
        model_type: str
    ) -> Path:
        """Save learned model with data-driven file naming"""

        # Generate domain name from path
        domain_name = Path(domain_path).name.lower().replace('-', '_')

        # Generate file name from discovered domain and model type
        file_name = self.file_namer.generate_domain_file_name(domain_path, model_type)

        # Determine organization based on content analysis
        if hasattr(model_data, 'statistical_analysis'):
            organization = self.organizer.determine_file_organization(model_data.statistical_analysis)
            file_path = Path(f"config/generated/learned_models/{organization['complexity_path']}/{organization['content_type_path']}/{file_name}")
        else:
            # Fallback to simple organization
            file_path = Path(f"config/generated/learned_models/standard/{file_name}")

        # Create directory structure
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(file_path, 'w') as f:
            if file_name.endswith('.json'):
                json.dump(model_data.model_dump(), f, indent=2)
            else:
                yaml.safe_dump(model_data.model_dump(), f, default_flow_style=False)

        return file_path

    async def save_validation_report(
        self,
        validation_result: ValidationResult,
        report_type: str
    ) -> Path:
        """Save validation report with timestamped naming"""

        # Generate timestamped file name
        file_name = self.file_namer.generate_timestamped_file_name(report_type)
        file_path = Path(f"config/generated/validation_reports/{file_name}")

        # Create directory and save
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(validation_result.model_dump(), f, indent=2)

        return file_path

    def discover_existing_models(self, domain_name: str) -> List[Path]:
        """Discover existing model files for a domain using data-driven search"""

        model_files = []
        base_path = Path("config/generated/learned_models")

        # Search in all complexity/content-type combinations
        for complexity_dir in base_path.iterdir():
            if complexity_dir.is_dir():
                for content_dir in complexity_dir.iterdir():
                    if content_dir.is_dir():
                        # Look for files matching domain pattern
                        pattern = f"{domain_name}_*.json"
                        model_files.extend(content_dir.glob(pattern))

        return model_files
```

### 5. Validation of Data-Driven Naming

```python
class DataDrivenNamingValidator:
    """Validate that file naming follows data-driven principles"""

    def validate_file_naming(self, file_path: Path) -> ValidationResult:
        """Validate that file names are data-driven, not hardcoded"""

        violations = []

        # Check for hardcoded domain names
        hardcoded_domains = ["programming", "language", "medical", "legal", "technical"]
        file_name = file_path.name.lower()

        for hardcoded_domain in hardcoded_domains:
            if file_name.startswith(hardcoded_domain + "_"):
                # Check if this domain actually exists in data/raw
                raw_domains = self._discover_actual_domains()
                if hardcoded_domain not in raw_domains:
                    violations.append(f"File name '{file_name}' contains hardcoded domain '{hardcoded_domain}' not found in data/raw")

        # Check for hardcoded timestamps
        if re.match(r'^\d{8}_\d{6}_', file_name):
            # This is good - timestamped file
            pass
        elif re.match(r'.*_\d{4}\d{2}\d{2}\.', file_name):
            violations.append(f"File name '{file_name}' appears to have hardcoded date instead of dynamic timestamp")

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            confidence=1.0 if len(violations) == 0 else 0.0
        )

    def _discover_actual_domains(self) -> List[str]:
        """Discover actual domains from data/raw directory"""
        raw_path = Path("data/raw")
        domains = []

        for item in raw_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                domain_name = item.name.lower().replace('-', '_')
                domains.append(domain_name)

        return domains
```

## Key Principles

### ✅ **Data-Driven File Naming:**
1. **Domain names** generated from subdirectory discovery
2. **File organization** based on content analysis
3. **Timestamps** generated dynamically
4. **Categories** determined from statistical analysis

### ✅ **Universal Design:**
1. **No hardcoded domain assumptions**
2. **Content-driven organization patterns**
3. **Scalable to any discovered domain**
4. **Adaptive to content characteristics**

### ✅ **Validation:**
1. **Automatic detection** of hardcoded names
2. **Verification** against actual discovered domains
3. **Compliance checking** with data-driven principles

This data-driven file naming strategy ensures that ALL file names and directory structures are generated from discovered data, maintaining the "Data-Driven Everything" principle throughout the configuration system.
