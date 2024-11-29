from transformers import AutoTokenizer

# Define base model checkpoint and fine-tuned model directory
base_model_checkpoint = "NousResearch/Llama-2-7b-chat-hf"
output_dir = "./finetuned_llama2"

# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint, trust_remote_code=True)

# Save the tokenizer to the fine-tuned model directory
tokenizer.save_pretrained(output_dir)

print(f"Tokenizer saved to {output_dir}")
