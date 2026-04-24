from pathlib import Path

for i in Path(".").glob("*.jpg"):
    if i.name.startswith("bsb"):
        new_name = i.name.split("_")[1]
        i.rename(i.parent / new_name)