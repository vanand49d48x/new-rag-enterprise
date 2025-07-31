#!/usr/bin/env python3
"""
Performance Monitoring for Advanced Model Optimizations
"""

import time
import psutil
import torch
import json
from typing import Dict, List

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def measure_inference_time(self, model, input_text: str, num_runs: int = 10):
        """Measure inference time"""
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        if tokenizer:
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(**inputs)
            
            # Measure
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(**inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            return {
                "avg_inference_time": avg_time,
                "min_time": min(times),
                "max_time": max(times),
                "std_dev": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            }
    
    def measure_memory_usage(self, model):
        """Measure memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        else:
            gpu_memory = 0
            gpu_memory_reserved = 0
        
        cpu_memory = psutil.virtual_memory().percent
        
        return {
            "gpu_memory_gb": gpu_memory,
            "gpu_memory_reserved_gb": gpu_memory_reserved,
            "cpu_memory_percent": cpu_memory
        }
    
    def measure_throughput(self, model, input_text: str, batch_size: int = 1):
        """Measure throughput (tokens/second)"""
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        if tokenizer:
            inputs = tokenizer(input_text, return_tensors="pt")
            if batch_size > 1:
                inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Count tokens
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = outputs.logits.shape[1] if hasattr(outputs, 'logits') else 0
            total_tokens = input_tokens + output_tokens
            
            throughput = total_tokens / total_time
            
            return {
                "throughput_tokens_per_sec": throughput,
                "total_tokens": total_tokens,
                "total_time": total_time
            }
    
    def generate_report(self, model_name: str, optimization_type: str):
        """Generate comprehensive performance report"""
        report = {
            "model_name": model_name,
            "optimization_type": optimization_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_cores": psutil.cpu_count(),
                "total_ram_gb": psutil.virtual_memory().total / 1024**3,
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # Test input
        test_input = "What is the capital of France?"
        
        # Measure performance
        report["inference_time"] = self.measure_inference_time(model, test_input)
        report["memory_usage"] = self.measure_memory_usage(model)
        report["throughput"] = self.measure_throughput(model, test_input)
        
        return report

def main():
    monitor = PerformanceMonitor()
    
    # Example usage
    print("📊 Performance Monitoring for Advanced Optimizations")
    print("Run this script after applying optimizations to measure performance gains")

if __name__ == "__main__":
    main()
