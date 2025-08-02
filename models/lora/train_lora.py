#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for RAG System
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json

def load_model_and_tokenizer(model_name):
    """Load base model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model, tokenizer

def create_lora_config():
    """Create LoRA configuration"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def prepare_training_data(data_path):
    """Prepare training data for LoRA fine-tuning"""
    # Load RAG-specific training data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Format for training
    formatted_data = []
    for item in data:
        formatted_data.append({
            "text": f"Context: {item.get('context', '')}\nQuestion: {item.get('question', '')}\nAnswer: {item.get('answer', '')}"
        })
    
    return Dataset.from_list(formatted_data)

def train_lora_model():
    """Main LoRA training function"""
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Apply LoRA
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Prepare training data
    dataset = prepare_training_data("./data/rag_training_data.json")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/lora/output",
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train the model
    trainer.train()
    
    # Save the LoRA model
    trainer.save_model("./models/lora/final_model")
    print("✅ LoRA training completed!")

if __name__ == "__main__":
    train_lora_model()
