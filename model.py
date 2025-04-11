from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

# Load pre-trained Pegasus model and tokenizer
model_name = "google/pegasus-cnn_dailymail"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Load your fine-tuning dataset
dataset = load_dataset("your_dataset")

# Tokenize and preprocess your dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_pegasus_model")
