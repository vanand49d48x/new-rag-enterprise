#!/usr/bin/env python3
"""
MoE (Mixture of Experts) Implementation for RAG System
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional

class MoEGate(nn.Module):
    """MoE Gate for expert routing"""
    def __init__(self, input_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(input_size, num_experts)
        
    def forward(self, x):
        gate_scores = self.gate(x)
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = torch.softmax(top_k_scores, dim=-1)
        return top_k_scores, top_k_indices

class Expert(nn.Module):
    """Individual expert in MoE"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class MoELayer(nn.Module):
    """MoE Layer with multiple experts"""
    def __init__(self, input_size: int, num_experts: int, expert_capacity: int, top_k: int = 2):
        super().__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.top_k = top_k
        
        self.gate = MoEGate(input_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(input_size, input_size * 2) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Get expert routing
        gate_scores, expert_indices = self.gate(x)
        
        # Reshape for expert processing
        x_flat = x.view(-1, hidden_size)
        gate_scores_flat = gate_scores.view(-1, self.top_k)
        expert_indices_flat = expert_indices.view(-1, self.top_k)
        
        # Process through experts
        expert_outputs = []
        for i in range(self.num_experts):
            # Find inputs for this expert
            expert_mask = (expert_indices_flat == i).any(dim=-1)
            if expert_mask.sum() > 0:
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[i](expert_input)
                expert_outputs.append((expert_output, expert_mask))
        
        # Combine expert outputs
        output = torch.zeros_like(x_flat)
        for expert_output, expert_mask in expert_outputs:
            output[expert_mask] += expert_output
        
        return output.view(batch_size, seq_len, hidden_size)

class MoEModel(nn.Module):
    """MoE Model for RAG system"""
    def __init__(self, base_model_name: str, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Add MoE layers
        hidden_size = self.base_model.config.hidden_size
        self.moe_layers = nn.ModuleList([
            MoELayer(hidden_size, num_experts, 64, top_k)
            for _ in range(4)  # Add MoE to 4 layers
        ])
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # Apply MoE layers
        hidden_states = outputs.hidden_states
        for i, moe_layer in enumerate(self.moe_layers):
            if i < len(hidden_states):
                hidden_states[i] = moe_layer(hidden_states[i])
        
        return outputs

def create_moe_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Create and return MoE model"""
    model = MoEModel(model_name)
    return model

if __name__ == "__main__":
    # Test MoE model creation
    model = create_moe_model()
    print("✅ MoE model created successfully!")
