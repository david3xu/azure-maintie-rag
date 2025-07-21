#!/usr/bin/env python3
"""
Simple script to convert .txt files to .md files in backend/data/raw.
Reads content, saves as .md, and deletes original .txt.
"""

import os
from pathlib import Path

def convert_txt_to_md(directory: str):
    """Converts all .txt files in a given directory to .md files.
    The original .txt files are deleted after conversion.
    """
    data_path = Path(directory)

    if not data_path.is_dir():
        print(f"Error: Directory not found at {data_path}")
        return

    print(f"Starting conversion of .txt to .md files in: {data_path}")
    converted_count = 0
    deleted_count = 0

    for file_path in data_path.iterdir():
        if file_path.is_file() and file_path.suffix == ".txt":
            md_file_path = file_path.with_suffix(".md")

            try:
                with open(file_path, 'r', encoding='utf-8') as f_txt:
                    content = f_txt.read()

                with open(md_file_path, 'w', encoding='utf-8') as f_md:
                    f_md.write(content)

                os.remove(file_path)
                print(f"Converted '{file_path.name}' to '{md_file_path.name}' and deleted original.")
                converted_count += 1
                deleted_count += 1

            except Exception as e:
                print(f"Error processing file {file_path.name}: {e}")

    print(f"\nConversion complete. Converted {converted_count} files, deleted {deleted_count} .txt originals.")

if __name__ == "__main__":
    # Assuming the script is run from the 'backend' directory or its parent
    # Adjust this path if the script's location or target directory changes
    target_data_directory = "data/raw"
    convert_txt_to_md(target_data_directory)