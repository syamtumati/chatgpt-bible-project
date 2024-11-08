from sklearn.model_selection import train_test_split
from pathlib import Path

# Path to the combined text file
combined_text_path = "data/sermons/bible_sermon_combined.txt"

# Read the combined file
with open(combined_text_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Split the data into 90% train and 10% eval
train_lines, eval_lines = train_test_split(lines, test_size=0.1, random_state=42)

# Save the train and eval splits
Path("data/sermons/bible_sermon_train.txt").write_text("".join(train_lines), encoding="utf-8")
Path("data/sermons/bible_sermon_eval.txt").write_text("".join(eval_lines), encoding="utf-8")

print("Dataset split into training and evaluation files.")
