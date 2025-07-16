"""Universal file utilities for any domain."""

from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import json
import yaml
import pickle
from datetime import datetime
import hashlib


class FileUtils:
    """Universal file utilities that work across all domains."""

    @staticmethod
    def ensure_directory(directory: str) -> Path:
        """Ensure directory exists and return Path object."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Read text file with error handling."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

    @staticmethod
    def write_text_file(file_path: str, content: str, encoding: str = 'utf-8') -> None:
        """Write text file with directory creation."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding=encoding) as f:
            f.write(content)

    @staticmethod
    def read_json_file(file_path: str) -> Dict[str, Any]:
        """Read JSON file with error handling."""
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def write_json_file(file_path: str, data: Any, indent: int = 2) -> None:
        """Write JSON file with directory creation."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)

    @staticmethod
    def read_yaml_file(file_path: str) -> Dict[str, Any]:
        """Read YAML file with error handling."""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def write_yaml_file(file_path: str, data: Dict[str, Any]) -> None:
        """Write YAML file with directory creation."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @staticmethod
    def read_pickle_file(file_path: str) -> Any:
        """Read pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def write_pickle_file(file_path: str, data: Any) -> None:
        """Write pickle file with directory creation."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def find_files(
        directory: str,
        pattern: str = "*",
        recursive: bool = True
    ) -> List[Path]:
        """Find files matching pattern."""
        path = Path(directory)

        if not path.exists():
            return []

        if recursive:
            return list(path.rglob(pattern))
        else:
            return list(path.glob(pattern))

    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
        """Get hash of file content."""
        hash_obj = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information."""
        path = Path(file_path)

        if not path.exists():
            return {'exists': False}

        stat = path.stat()

        return {
            'exists': True,
            'path': str(path.absolute()),
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'is_file': path.is_file(),
            'is_directory': path.is_dir(),
            'hash_md5': FileUtils.get_file_hash(file_path) if path.is_file() else None
        }

    @staticmethod
    def batch_process_files(
        file_paths: List[str],
        processor_func,
        batch_size: int = 10
    ) -> Generator[List[Any], None, None]:
        """Process files in batches."""
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batch_results = []

            for file_path in batch:
                try:
                    result = processor_func(file_path)
                    batch_results.append(result)
                except Exception as e:
                    batch_results.append({'error': str(e), 'file': file_path})

            yield batch_results

    @staticmethod
    def safe_filename(name: str, max_length: int = 255) -> str:
        """Create safe filename from string."""
        # Replace invalid characters
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")

        # Replace spaces with underscores
        safe_name = safe_name.replace(" ", "_")

        # Remove multiple underscores
        while "__" in safe_name:
            safe_name = safe_name.replace("__", "_")

        # Limit length
        if len(safe_name) > max_length:
            safe_name = safe_name[:max_length]

        # Ensure not empty
        if not safe_name:
            safe_name = "unnamed"

        return safe_name

    @staticmethod
    def create_backup(file_path: str, backup_suffix: str = ".bak") -> str:
        """Create backup of file."""
        backup_path = f"{file_path}{backup_suffix}"

        if Path(file_path).exists():
            content = FileUtils.read_text_file(file_path)
            FileUtils.write_text_file(backup_path, content)

        return backup_path