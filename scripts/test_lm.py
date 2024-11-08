from transformers import RobertaForMaskedLM, RobertaTokenizerFast
import torch

# Load the tokenizer
#tokenizer = RobertaTokenizerFast.from_pretrained("data/processed/tokenizer")

# Load the tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained("models/Bible-project")
model = RobertaForMaskedLM.from_pretrained("models/Bible-project")

# Check tokenizer details for [MASK] token
#print("Mask token ID:", tokenizer.mask_token_id)
#print("Vocabulary size:", tokenizer.vocab_size)

# Check if tokenizer recognizes the [MASK] token
#if "[MASK]" not in tokenizer.get_vocab():
#    print("The tokenizer does not recognize the [MASK] token.")
#else:
#    print("The tokenizer recognizes the [MASK] token.")

# Ensure <mask> token is recognized
if "<mask>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    model.resize_token_embeddings(len(tokenizer))
    
# Confirm mask token ID
print("Mask token ID:", tokenizer.mask_token_id)
print("Vocabulary size:", tokenizer.vocab_size)


# Load the model (no extra arguments needed)
#model = RobertaForMaskedLM.from_pretrained("models/Bible-project")

# Test with a sample sentence with a masked token
#sample_text = "God is [MASK]"

# Test with a sample sentence containing the <mask> token
sample_text = "God is <mask>."

# Tokenize the input text
inputs = tokenizer(sample_text, return_tensors="pt")
print("Tokenized input:", inputs)

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Identify the index of the [MASK] token
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

# Predict the token for [MASK]
predicted_token_id = logits[0, mask_token_index, :].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

#print("Predicted word for [MASK]:", predicted_token)

print("Predicted word for <mask>:", predicted_token)
