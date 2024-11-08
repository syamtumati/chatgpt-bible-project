from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaConfig

# Load the ByteLevelBPETokenizer from vocab and merges files
tokenizer = ByteLevelBPETokenizer(
    "data/processed/tokenizer/vocab.json",
    "data/processed/tokenizer/merges.txt"
)

# Get the vocabulary size from the tokenizer
vocab_size = tokenizer.get_vocab_size()

# Configure the RobertaConfig with the tokenizer's vocab size
config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Save the configuration
config.save_pretrained("models/Bible-project")
print("Configuration saved successfully.")
