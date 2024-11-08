from transformers import RobertaForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import RobertaTokenizerFast
from datasets import load_dataset
import os

# Load the tokenizer
#tokenizer = RobertaTokenizerFast.from_pretrained("data/processed/tokenizer")

# Ensure the output directory exists
os.makedirs("models/Bible-project", exist_ok=True)

# Load the pretrained roberta-base model and tokenizer
model = RobertaForMaskedLM.from_pretrained("roberta-base")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Load your dataset
dataset = load_dataset('text', data_files={'train': 'data/sermons/bible_sermon_train.txt', 'validation': 'data/sermons/bible_sermon_eval.txt'})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up data collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Load model configuration and initialize model
#model = RobertaForMaskedLM.from_pretrained("models/Bible-project", config="models/Bible-project/config.json")

# Define training arguments
training_args = TrainingArguments(
    output_dir="models/Bible-project",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    learning_rate=5e-5
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model after training
trainer.save_model("models/Bible-project")
tokenizer.save_pretrained("models/Bible-project")

