# This script cleans up file and folder names to match the genre naming convention

# Call this as follows:
# python3 rename_items.py /path/to/your/folder --dry-run
# remove the --dry-run flag to actually perform the changes

import os
from pathlib import Path

# Mapping of original text to replacement
REPLACEMENTS = {
    'static1': 'sta01',
    'static01': 'sta01',
    'static2': 'sta02',
    'static02': 'sta02',
    'static03': 'sta03',
    'static3': 'sta03',
    'worldA09': 'woA09',
    'worldA9': 'woA09',
    'worldA13': 'woA13',
    'worldB02': 'woB02',
    'worldB2': 'woB02',
    'worldB04': 'woB04',
    'worldB4': 'woB04',
    'worldB05': 'woB05',
    'worldB5': 'woB05',
    'worldC07': 'woC07',
    'worldC7': 'woC07',
    'worldC08': 'woC08',
    'worldC8': 'woC08',
    'worldC10': 'woC10',
    'worldC13': 'woC13',
    'worldC14': 'woC14',
    'worldA': 'woA',
    'worldB': 'woB',
    'worldC': 'woC',
    'util01': 'uti01',
    'util02': 'uti02',
    'util03': 'uti03',
    'util3': 'uti03',
    'inl01': 'ind01',
    'inl02': 'ind02',
    'inl03': 'ind03',
    'inl06': 'ind06',
    'inl07': 'ind07',
    'inl08': 'ind08',
    'inl11': 'ind11',
    'inr04': 'ind04',
    'inr05': 'ind05',
    'inr09': 'ind09',
    'inr10': 'ind10',
    'FOLK': 'folk'
}

def apply_replacements(name):
    for old, new in REPLACEMENTS.items():
        if old in name:
            name = name.replace(old, new)
    return name

def rename_items(root_dir: Path, dry_run: bool = False):
    root_dir = Path(root_dir).resolve()

    # Process child paths before parent paths (deepest first)
    for path in sorted(root_dir.rglob("*"), key=lambda p: -p.as_posix().count("/")):
        new_name = apply_replacements(path.name)
        if new_name != path.name:
            new_path = path.with_name(new_name)
            if dry_run:
                print(f"[DRY-RUN] Would rename: {path} → {new_path}")
            else:
                print(f"Renaming: {path} → {new_path}")
                try:
                    path.rename(new_path)
                except Exception as e:
                    print(f"Failed to rename {path}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch rename files and folders using custom rules.")
    parser.add_argument("root", type=str, nargs="?", default=".", help="Root directory to search (default: current dir)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without renaming")
    args = parser.parse_args()

    rename_items(Path(args.root), dry_run=args.dry_run)
