from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load the tokenizer and model
model_name_or_path = "./fine-tuned-gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Define a route for generating text
@app.route('/generate', methods=['POST'])
def generate_text():
    # Get the input text from the request
    input_data = request.json
    input_text = input_data.get('input_text', '')

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Create an attention mask
    attention_mask = input_ids != tokenizer.pad_token_id

    # Generate text with attention mask
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the generated text as a JSON response
    return jsonify({'generated_text': generated_text})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='ip_address', port=5000)
