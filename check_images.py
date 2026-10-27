
from pathlib import Path
from PIL import Image

root = Path("data")
bad_files = []
checked = 0

for split in ["train", "val", "test"]:
    split_path = root / split

    for path in split_path.rglob("*"):
        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                img.verify()
            checked += 1
        except Exception as error:
            bad_files.append((str(path), str(error)))

print(f"Valid images checked: {checked}")
print(f"Unreadable files found: {len(bad_files)}")

with open("bad_images.txt", "w", encoding="utf-8") as file:
    for path, error in bad_files:
        file.write(f"{path} | {error}\n")

print("Full list saved to bad_images.txt")

for path, error in bad_files[:20]:
    print(path, "|", error)
