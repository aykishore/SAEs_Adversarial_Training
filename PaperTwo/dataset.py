from pathlib import Path

mery_root = Path("data/FERG_DB/FERG_DB_256/mery")

joy_dir = mery_root / "mery_joy"
neutral_dir = mery_root / "mery_neutral"

joy_images = list(joy_dir.glob("*"))
neutral_images = list(neutral_dir.glob("*"))

print("Joy images:", len(joy_images))
print("Neutral images:", len(neutral_images))

print("Example joy image:", joy_images[0])
print("Example neutral image:", neutral_images[0])