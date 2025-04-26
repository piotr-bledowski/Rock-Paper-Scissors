import os
import pandas as pd
import shutil

KHULAN_PATH = "data/khulan"
SARANJIL_PATH = "data/Saranjil"
DATA_PATH = "data"
SHIFT_AMOUNT = 268  

os.makedirs(DATA_PATH, exist_ok=True)

for filename in os.listdir(KHULAN_PATH):
    if filename.endswith(".jpg"):
        shutil.copy(os.path.join(KHULAN_PATH, filename), os.path.join(DATA_PATH, filename))

for filename in os.listdir(SARANJIL_PATH):
    if filename.endswith(".jpg"):
        original_index = int(filename.replace(".jpg", ""))
        new_index = original_index + SHIFT_AMOUNT
        new_filename = f"{new_index}.jpg"
        shutil.copy(os.path.join(SARANJIL_PATH, filename), os.path.join(DATA_PATH, new_filename))

khulan_df = pd.read_csv(os.path.join(KHULAN_PATH, "annotations.csv"), header=None, names=["index", "label"])
saranjil_df = pd.read_csv(os.path.join(SARANJIL_PATH, "annotations.csv"), header=None, names=["index", "label"])
saranjil_df["index"] += SHIFT_AMOUNT  

merged_df = pd.concat([khulan_df, saranjil_df])
merged_df.to_csv(os.path.join(DATA_PATH, "annotations.csv"), index=False, header=False)

print(" Dataset prepared successfully! All files are in 'data/' and annotations are merged.")
