from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Dataset location
# ==========================================

DATASET_DIR = Path("PlantVillage-Dataset/raw/color")

# Output directory
OUTPUT_DIR = Path("dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. Collect image paths and labels
# ==========================================

print("Scanning dataset...")

data = []

valid_extensions = {".jpg", ".jpeg", ".png"}

for class_dir in sorted(DATASET_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for image_path in class_dir.iterdir():

        if image_path.suffix.lower() in valid_extensions:

            data.append({
                "image_path": str(image_path),
                "label": class_name
            })

df = pd.DataFrame(data)

print(f"\nTotal images found: {len(df)}")
print(f"Total classes found: {df['label'].nunique()}")

# ==========================================
# 3. Create 80% train / 10% validation / 10% test
# ==========================================

print("\nCreating dataset splits...")

# First: 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# Second: split temporary 50/50
# 10% validation + 10% test
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# ==========================================
# 4. Save CSV files
# ==========================================

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# ==========================================
# 5. Display results
# ==========================================

print("\nDataset split completed!")

print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(validation_df)}")
print(f"Testing images:    {len(test_df)}")

print("\nCSV files created:")
print("dataset/train.csv")
print("dataset/validation.csv")
print("dataset/test.csv")

print("\nClass distribution:")
print(train_df["label"].value_counts().sort_index())