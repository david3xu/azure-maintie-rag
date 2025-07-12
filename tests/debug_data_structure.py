import json
from pathlib import Path
from config.settings import settings

def inspect_maintie_data():
    """Inspect the actual structure of your MaintIE JSON files"""

    data_files = [
        settings.raw_data_dir / settings.gold_data_filename,
        settings.raw_data_dir / settings.silver_data_filename
    ]

    for file_path in data_files:
        if file_path.exists():
            print(f"\n📁 Inspecting: {file_path}")

            with open(file_path, 'r') as f:
                data = json.load(f)

            print(f"📊 Data type: {type(data)}")
            print(f"📊 Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

            # Sample first item
            if isinstance(data, list) and data:
                print(f"📊 First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                print(f"📊 Sample item: {json.dumps(data[0], indent=2)[:500]}...")
            elif isinstance(data, dict):
                for key, value in list(data.items())[:3]:
                    print(f"📊 {key}: {type(value)} - {str(value)[:100]}...")
        else:
            print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    inspect_maintie_data()
