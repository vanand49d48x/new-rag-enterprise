#!/usr/bin/env python3
"""
Advanced Model Optimization Pipeline
Implements NF4/4BIT quantization, LoRA, MoE, and Double Quantization
"""

import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import argparse

class AdvancedOptimizer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def apply_double_quantization(self, model_name: str):
        """Apply double quantization with nested quantization layers"""
        print(f"🔧 Applying Double Quantization to {model_name}")
        
        # Create double quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Enable double quantization
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,  # Additional storage optimization
        )
        
        # Load model with double quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            # Additional memory optimizations
            low_cpu_mem_usage=True,
            offload_folder="./offload",
            # Enable gradient checkpointing for memory efficiency
            use_gradient_checkpointing=True,
        )
        
        # Apply additional memory optimizations
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        print(f"✅ Double quantization applied successfully")
        print(f"   - 4-bit NF4 quantization enabled")
        print(f"   - Double quantization enabled")
        print(f"   - Memory optimizations applied")
        
        return model
    
    def apply_nf4_quantization(self, model_name: str):
        """Apply NF4 quantization to model"""
        print(f"🔧 Applying NF4 quantization to {model_name}")
        
        # Create quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with NF4 quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config
        )
        
        return model
    
    def apply_lora_finetuning(self, model, lora_config_path: str):
        """Apply LoRA fine-tuning"""
        print("🎯 Applying LoRA fine-tuning")
        
        with open(lora_config_path, 'r') as f:
            lora_config = yaml.safe_load(f)['lora']
        
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, peft_config)
        return model
    
    def apply_moe_architecture(self, model_name: str):
        """Apply MoE architecture"""
        print("🧠 Applying MoE architecture")
        
        # Import MoE implementation
        import sys
        sys.path.append('./models/moe')
        from moe_model import create_moe_model
        
        model = create_moe_model(model_name)
        return model
    
    def optimize_model(self, model_name: str, optimization_type: str):
        """Main optimization function"""
        print(f"🚀 Starting {optimization_type} optimization for {model_name}")
        
        if optimization_type == "double_quant":
            model = self.apply_double_quantization(model_name)
        elif optimization_type == "nf4":
            model = self.apply_nf4_quantization(model_name)
        elif optimization_type == "lora":
            base_model = AutoModelForCausalLM.from_pretrained(model_name)
            model = self.apply_lora_finetuning(base_model, "configs/advanced/lora_config.yaml")
        elif optimization_type == "moe":
            model = self.apply_moe_architecture(model_name)
        elif optimization_type == "combined":
            # Apply all optimizations
            model = self.apply_double_quantization(model_name)
            model = self.apply_lora_finetuning(model, "configs/advanced/lora_config.yaml")
            # Note: MoE requires special architecture, not easily combined
        
        return model

def main():
    parser = argparse.ArgumentParser(description="Advanced Model Optimization")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model to optimize")
    parser.add_argument("--optimization", type=str, choices=["double_quant", "nf4", "lora", "moe", "combined"], help="Optimization type")
    parser.add_argument("--config", type=str, default="configs/advanced/double_quant_config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    optimizer = AdvancedOptimizer(args.config)
    model = optimizer.optimize_model(args.model, args.optimization)
    
    # Save optimized model
    output_path = f"models/optimized/{args.optimization}_model"
    model.save_pretrained(output_path)
    print(f"✅ Optimized model saved to {output_path}")

if __name__ == "__main__":
    main()
