from transformers import RobertaConfig, RobertaForMaskedLM
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# Load the configuration
config = RobertaConfig.from_pretrained("models/Bible-project")

# Initialize a new model from scratch with this configuration
model = RobertaForMaskedLM(config)

# Save the initialized model
model.save_pretrained("models/Bible-project")

# Load the tokenizer
tokenizer = ByteLevelBPETokenizer(
    "data/processed/tokenizer/vocab.json",
    "data/processed/tokenizer/merges.txt"
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print("Model and tokenizer initialized successfully.")
