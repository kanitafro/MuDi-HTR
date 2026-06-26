from pathlib import Path

# Get root directory of the repository
repo_root = Path(__file__).parent.parent
isgl_dir = repo_root / "data" / "raw" / "ISGL"
txt_files = list(isgl_dir.rglob("*.txt"))
print(f"Found {len(txt_files)} .txt files")
for f in txt_files[:10]:
    print(f)

print("\nFull path of the searched directory:", isgl_dir.resolve())