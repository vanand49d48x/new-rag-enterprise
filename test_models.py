#!/usr/bin/env python3
"""
Comprehensive Model Testing for C7 Paperspace
Tests both Qwen2.5-3B and Qwen2-7B with all advanced optimizations
"""

import time
import json
import requests
import subprocess
import sys
from typing import Dict, List

class ModelTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.llm_url = "http://localhost:8080"
        
    def test_model_switch(self, model_name: str):
        """Switch to specified model and test"""
        print(f"🔄 Switching to {model_name}...")
        
        try:
            # Switch model
            result = subprocess.run(["./switch_model.sh", model_name], 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # Wait for services to restart
            print("⏳ Waiting for services to restart...")
            time.sleep(60)
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to switch model: {e}")
            return False
    
    def test_llm_inference(self, prompt: str, expected_tokens: int = 50):
        """Test LLM inference performance"""
        print(f"🧠 Testing LLM inference...")
        
        payload = {
            "prompt": prompt,
            "n_predict": expected_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.llm_url}/completion",
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                inference_time = end_time - start_time
                tokens_generated = result.get('tokens_predicted', 0)
                tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0
                
                print(f"✅ Inference successful!")
                print(f"   ⏱️  Time: {inference_time:.2f}s")
                print(f"   📝 Tokens: {tokens_generated}")
                print(f"   🚀 Speed: {tokens_per_sec:.1f} tokens/sec")
                print(f"   💬 Response: {result.get('content', '')[:100]}...")
                
                return {
                    "success": True,
                    "inference_time": inference_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_sec": tokens_per_sec
                }
            else:
                print(f"❌ LLM inference failed: {response.status_code}")
                return {"success": False}
                
        except Exception as e:
            print(f"❌ LLM inference error: {str(e)}")
            return {"success": False}
    
    def test_rag_pipeline(self, query: str):
        """Test RAG pipeline performance"""
        print(f"🔍 Testing RAG pipeline...")
        
        payload = {"query": query}
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                rag_time = end_time - start_time
                
                print(f"✅ RAG pipeline successful!")
                print(f"   ⏱️  Time: {rag_time:.2f}s")
                print(f"   📊 Sources: {len(result.get('sources', []))}")
                print(f"   💬 Answer: {result.get('answer', '')[:200]}...")
                
                return {
                    "success": True,
                    "rag_time": rag_time,
                    "sources_count": len(result.get('sources', []))
                }
            else:
                print(f"❌ RAG pipeline failed: {response.status_code}")
                return {"success": False}
                
        except Exception as e:
            print(f"❌ RAG pipeline error: {str(e)}")
            return {"success": False}
    
    def test_system_resources(self):
        """Test system resource usage"""
        print("💻 Testing system resources...")
        
        try:
            # Get Docker stats
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
                capture_output=True, text=True, check=True
            )
            
            print("📊 Docker Container Stats:")
            print(result.stdout)
            
            return True
        except Exception as e:
            print(f"❌ Failed to get system stats: {str(e)}")
            return False
    
    def run_comprehensive_test(self, model_name: str):
        """Run comprehensive test for a model"""
        print(f"\n{'='*60}")
        print(f"🧪 COMPREHENSIVE TEST: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Test prompts
        llm_prompts = [
            "Explain quantum computing in simple terms.",
            "What are the main benefits of renewable energy?",
            "How does machine learning work?"
        ]
        
        rag_queries = [
            "What is the capital of France?",
            "Explain the symptoms of diabetes.",
            "What are the latest developments in AI?"
        ]
        
        # Test LLM inference
        print("\n🧠 LLM Inference Tests:")
        llm_results = []
        for i, prompt in enumerate(llm_prompts, 1):
            print(f"\n--- Test {i} ---")
            result = self.test_llm_inference(prompt)
            llm_results.append(result)
        
        # Test RAG pipeline
        print("\n🔍 RAG Pipeline Tests:")
        rag_results = []
        for i, query in enumerate(rag_queries, 1):
            print(f"\n--- Test {i} ---")
            result = self.test_rag_pipeline(query)
            rag_results.append(result)
        
        # System resources
        print("\n💻 System Resources:")
        self.test_system_resources()
        
        # Summary
        print(f"\n📊 {model_name.upper()} TEST SUMMARY:")
        successful_llm = sum(1 for r in llm_results if r.get('success', False))
        successful_rag = sum(1 for r in rag_results if r.get('success', False))
        
        if successful_llm > 0:
            avg_llm_time = sum(r.get('inference_time', 0) for r in llm_results if r.get('success', False)) / successful_llm
            avg_tokens_per_sec = sum(r.get('tokens_per_sec', 0) for r in llm_results if r.get('success', False)) / successful_llm
            print(f"   🧠 LLM: {successful_llm}/3 successful, avg {avg_llm_time:.2f}s, {avg_tokens_per_sec:.1f} tokens/sec")
        
        if successful_rag > 0:
            avg_rag_time = sum(r.get('rag_time', 0) for r in rag_results if r.get('success', False)) / successful_rag
            print(f"   🔍 RAG: {successful_rag}/3 successful, avg {avg_rag_time:.2f}s")
        
        return {
            "model": model_name,
            "llm_results": llm_results,
            "rag_results": rag_results
        }
    
    def run_all_tests(self):
        """Run tests for both models"""
        print("🚀 C7 PAPERSACE MODEL TESTING")
        print("Testing both Qwen2.5-3B and Qwen2-7B with all optimizations")
        print("="*60)
        
        models = ["qwen25_3b", "qwen2_7b"]
        all_results = []
        
        for model in models:
            # Switch to model
            if self.test_model_switch(model):
                # Run comprehensive test
                results = self.run_comprehensive_test(model)
                all_results.append(results)
                
                # Wait between tests
                print("\n⏳ Waiting 30 seconds before next test...")
                time.sleep(30)
            else:
                print(f"❌ Failed to switch to {model}")
        
        # Final comparison
        print(f"\n{'='*60}")
        print("📊 FINAL COMPARISON")
        print(f"{'='*60}")
        
        for result in all_results:
            model = result['model']
            llm_results = result['llm_results']
            rag_results = result['rag_results']
            
            successful_llm = sum(1 for r in llm_results if r.get('success', False))
            successful_rag = sum(1 for r in rag_results if r.get('success', False))
            
            if successful_llm > 0:
                avg_llm_time = sum(r.get('inference_time', 0) for r in llm_results if r.get('success', False)) / successful_llm
                avg_tokens_per_sec = sum(r.get('tokens_per_sec', 0) for r in llm_results if r.get('success', False)) / successful_llm
                print(f"\n{model.upper()}:")
                print(f"   🧠 LLM: {avg_llm_time:.2f}s avg, {avg_tokens_per_sec:.1f} tokens/sec")
                print(f"   🔍 RAG: {successful_rag}/3 successful")
        
        print(f"\n✅ Testing complete!")

def main():
    tester = ModelTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 