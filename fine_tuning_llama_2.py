#Import necessary modules
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import os

#Replace with preferred cache location
os.environ["HF_HOME"] = "/path/to/huggingface_cache"

#Load WikiText Dataset
def load_wikitext():
    print("Loading WikiText dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    return dataset

#Preprocess the Dataset
def preprocess_data(dataset, tokenizer, block_size=512):
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        return result

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized.map(group_texts, batched=True)

#Fine-Tune the Model with LoRA
def fine_tune_model(tokenizer, train_dataset, eval_dataset, model_checkpoint, output_dir):
    print("Loading model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # Disabling cache for fine-tuning
    model.config.use_cache = False 

    # Define LoRA configuration
    peft_args = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Fine-tune with SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_args,
        args=training_args,
        max_seq_length=block_size,
    )

    print("Starting training...")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    print("Model fine-tuned and saved.")
    return trainer.model

# Main
if __name__ == "__main__":

    # Configurations of LLaMA-2 model
    model_checkpoint = "NousResearch/Llama-2-7b-chat-hf" 
    output_dir = "./finetuned_llama2"
    block_size = 512

    # Loading and preprocessing data
    dataset = load_wikitext()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    tokenized_dataset = preprocess_data(dataset, tokenizer, block_size)

    # Splitting into train and eval datasets
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    # Fine-tuning the model
    finetuned_model = fine_tune_model(tokenizer, train_dataset, eval_dataset, model_checkpoint, output_dir)

    # Loading the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    finetuned_model = AutoModelForCausalLM.from_pretrained(output_dir).to("cuda" if torch.cuda.is_available() else "cpu")
