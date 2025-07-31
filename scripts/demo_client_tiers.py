#!/usr/bin/env python3
"""
Client AI Solutions Demo Script
Demonstrates different tiers of local AI solutions
"""

import time
import json
import requests
from typing import Dict, List

class AISolutionDemo:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        
    def demo_tier_1_laptop_mode(self):
        """Demo Tier 1: Laptop Mode - Basic AI"""
        print("🖥️  TIER 1: LAPTOP MODE (Basic)")
        print("=" * 50)
        print("Perfect for: Solo founders, testing, small teams")
        print("Hardware: Any laptop with 8GB+ RAM")
        print("Model: Phi-2 (2.7B parameters)")
        print("Performance: 2-5 seconds response time")
        print()
        
        # Simulate basic query
        query = "What are the main symptoms of diabetes?"
        print(f"Query: {query}")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"query": query},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {result['answer'][:200]}...")
                print(f"📊 Sources: {len(result['sources'])} documents")
                print(f"⚡ Search Method: {result['metadata']['search_method']}")
            else:
                print("❌ Demo unavailable (system starting up)")
        except Exception as e:
            print(f"❌ Demo unavailable: {str(e)}")
        
        print("\n" + "=" * 50)

    def demo_tier_2_workstation_mode(self):
        """Demo Tier 2: Workstation Mode - Standard AI"""
        print("⚙️  TIER 2: WORKSTATION MODE (Standard)")
        print("=" * 50)
        print("Perfect for: Startups, SMEs, growing teams")
        print("Hardware: Mac M2/M3 or PC with 16GB+ RAM")
        print("Model: Qwen2-7B (quantized)")
        print("Performance: 3-8 seconds response time")
        print()
        
        # Simulate complex multi-source query
        query = "What are the symptoms, treatments, and medications for diabetes?"
        print(f"Query: {query}")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"query": query},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {result['answer'][:300]}...")
                print(f"📊 Sources: {len(result['sources'])} documents")
                print(f"⚡ Search Method: {result['metadata']['search_method']}")
                print(f"🔍 Retrieved Chunks: {result['metadata']['retrieved_chunks']}")
            else:
                print("❌ Demo unavailable (system starting up)")
        except Exception as e:
            print(f"❌ Demo unavailable: {str(e)}")
        
        print("\n" + "=" * 50)

    def demo_tier_3_server_mode(self):
        """Demo Tier 3: Server Mode - Professional AI"""
        print("🖥️  TIER 3: SERVER MODE (Professional)")
        print("=" * 50)
        print("Perfect for: Mid-size teams, enterprise deployments")
        print("Hardware: Dedicated server with GPU (RTX 4070/4080)")
        print("Model: Qwen2-13B or Mixtral-8x7B (quantized)")
        print("Performance: 1-3 seconds response time")
        print()
        
        # Simulate enterprise query
        query = "Compare the side effects and interactions of diabetes medications across all available sources"
        print(f"Query: {query}")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"query": query},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {result['answer'][:400]}...")
                print(f"📊 Sources: {len(result['sources'])} documents")
                print(f"⚡ Search Method: {result['metadata']['search_method']}")
                print(f"🔍 Retrieved Chunks: {result['metadata']['retrieved_chunks']}")
            else:
                print("❌ Demo unavailable (system starting up)")
        except Exception as e:
            print(f"❌ Demo unavailable: {str(e)}")
        
        print("\n" + "=" * 50)

    def demo_tier_4_enterprise_mode(self):
        """Demo Tier 4: Enterprise Mode - Premium AI"""
        print("🏢 TIER 4: ENTERPRISE MODE (Premium)")
        print("=" * 50)
        print("Perfect for: Large organizations, security-focused deployments")
        print("Hardware: On-premises GPU server (RTX 4090, A100)")
        print("Model: Qwen2-72B or Llama2-70B (quantized)")
        print("Performance: <1 second response time")
        print()
        
        # Simulate critical business query
        query = "Analyze all medical documents and provide a comprehensive risk assessment for diabetes treatment protocols, including medication interactions, patient monitoring requirements, and emergency protocols"
        print(f"Query: {query}")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"query": query},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {result['answer'][:500]}...")
                print(f"📊 Sources: {len(result['sources'])} documents")
                print(f"⚡ Search Method: {result['metadata']['search_method']}")
                print(f"🔍 Retrieved Chunks: {result['metadata']['retrieved_chunks']}")
            else:
                print("❌ Demo unavailable (system starting up)")
        except Exception as e:
            print(f"❌ Demo unavailable: {str(e)}")
        
        print("\n" + "=" * 50)

    def show_cost_comparison(self):
        """Show cost comparison between local AI and cloud APIs"""
        print("💰 COST COMPARISON: Local AI vs Cloud APIs")
        print("=" * 50)
        
        # Monthly usage estimates
        usage_scenarios = {
            "Small Team (100 queries/day)": {
                "cloud_api": "$150-300/month",
                "local_ai": "$0/month (after hardware)",
                "savings": "100%"
            },
            "Medium Team (500 queries/day)": {
                "cloud_api": "$750-1500/month",
                "local_ai": "$0/month (after hardware)",
                "savings": "100%"
            },
            "Large Team (2000 queries/day)": {
                "cloud_api": "$3000-6000/month",
                "local_ai": "$0/month (after hardware)",
                "savings": "100%"
            }
        }
        
        for scenario, costs in usage_scenarios.items():
            print(f"\n📊 {scenario}")
            print(f"   Cloud API: {costs['cloud_api']}")
            print(f"   Local AI: {costs['local_ai']}")
            print(f"   Savings: {costs['savings']}")
        
        print("\n💡 Hardware Investment (one-time):")
        print("   Laptop Mode: $0 (use existing)")
        print("   Workstation Mode: $500-2000")
        print("   Server Mode: $2000-5000")
        print("   Enterprise Mode: $5000-20000")
        
        print("\n" + "=" * 50)

    def show_security_benefits(self):
        """Show security and privacy benefits"""
        print("🔒 SECURITY & PRIVACY BENEFITS")
        print("=" * 50)
        
        benefits = [
            "✅ Zero data leaves your network",
            "✅ No vendor lock-in",
            "✅ Complete data sovereignty",
            "✅ No API rate limits",
            "✅ No internet dependency",
            "✅ Customizable security policies",
            "✅ Audit trail control",
            "✅ Compliance ready (GDPR, HIPAA, etc.)"
        ]
        
        for benefit in benefits:
            print(f"   {benefit}")
        
        print("\n" + "=" * 50)

    def run_full_demo(self):
        """Run the complete demo for clients"""
        print("🚀 LOCAL AI SOLUTIONS DEMO")
        print("=" * 60)
        print("Demonstrating different tiers of local AI solutions")
        print("Current system: Tier 2 (Workstation Mode)")
        print("=" * 60)
        print()
        
        # Run tier demonstrations
        self.demo_tier_1_laptop_mode()
        self.demo_tier_2_workstation_mode()
        self.demo_tier_3_server_mode()
        self.demo_tier_4_enterprise_mode()
        
        # Show benefits
        self.show_cost_comparison()
        self.show_security_benefits()
        
        print("🎯 NEXT STEPS FOR CLIENTS:")
        print("=" * 50)
        print("1. Choose your tier based on needs and budget")
        print("2. We'll set up your local AI solution")
        print("3. Train your team on the new system")
        print("4. Enjoy unlimited AI without ongoing costs!")
        print()
        print("💬 Ready to transform your business with local AI?")

if __name__ == "__main__":
    demo = AISolutionDemo()
    demo.run_full_demo() 