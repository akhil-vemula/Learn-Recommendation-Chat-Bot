from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load the tokenizer and model
model_name_or_path = "path/ai_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Set the padding token to be the same as the EOS token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Resize the model's token embeddings to accommodate the padding token
model.resize_token_embeddings(len(tokenizer))

# Specify the path to your JSON file
json_file_path = "path/courses.json"

# Load data from the JSON file
with open(json_file_path, 'r') as f:
    courses_data = json.load(f)

# Prepare the text data for the model
text_data = [f"{course['course_title']}: {course['course_description']}" for course in courses_data]

# Create a Dataset object
dataset = Dataset.from_dict({"text": text_data})

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the input text
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # Use input_ids as labels
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save checkpoints and logs
    evaluation_strategy="no",  # Disable evaluation for simplicity
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Set batch size to 1
    num_train_epochs=5,  # Increase epochs for better learning
    weight_decay=0.01,
    save_steps=10,  # Save model every 10 steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model_save_path = "./fine-tuned-ai_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
