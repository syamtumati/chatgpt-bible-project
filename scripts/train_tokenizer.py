from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

# Paths
data_path = Path("data/sermons/bible_sermon_combined.txt")
tokenizer_path = Path("data/processed/tokenizer/")

# Initialize and train the tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[str(data_path)], vocab_size=52000, min_frequency=2, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>", "[MASK]"
])

# Save the tokenizer
tokenizer.save_model(str(tokenizer_path))