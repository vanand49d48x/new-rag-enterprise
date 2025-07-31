#!/usr/bin/env python3
"""
Medical RAG Test
Comprehensive test of medical RAG functionality
"""

import requests
import time
import json
import sys

API_BASE = "http://localhost:8000"

def test_medical_rag():
    """Test medical RAG functionality"""
    print("🏥 Medical RAG Test")
    print("=" * 50)
    
    # Test medical queries
    medical_queries = [
        "What are the symptoms of diabetes?",
        "What medications treat hypertension?",
        "What are asthma emergency signs?",
        "What are depression symptoms?",
        "What are metformin side effects?",
        "How to manage diabetes and hypertension together?",
        "What are the warning signs of heart attack?",
        "Compare SSRIs and SNRIs for depression"
    ]
    
    results = []
    
    for i, query in enumerate(medical_queries, 1):
        print(f"\n========== Test {i}/{len(medical_queries)} ==========")
        print(f"🔍 Query: {query}")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_BASE}/chat",
                json={"query": query, "top_k": 5},
                timeout=180  # Extended timeout for medical queries
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                sources = result.get('sources', [])
                
                print(f"✅ Success in {response_time:.2f}s")
                print(f"📝 Answer: {answer[:200]}...")
                print(f"📚 Sources: {len(sources)}")
                
                results.append({
                    "query": query,
                    "response_time": response_time,
                    "answer_length": len(answer),
                    "sources_count": len(sources),
                    "success": True
                })
                
            else:
                print(f"❌ Failed: HTTP {response.status_code}")
                results.append({
                    "query": query,
                    "response_time": response_time,
                    "success": False
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "query": query,
                "response_time": time.time() - start_time,
                "success": False
            })
        
        # Wait between tests
        if i < len(medical_queries):
            print("⏳ Waiting 3 seconds...")
            time.sleep(3)
    
    # Calculate metrics
    successful_tests = [r for r in results if r.get('success', False)]
    
    if successful_tests:
        response_times = [r['response_time'] for r in successful_tests]
        answer_lengths = [r['answer_length'] for r in successful_tests]
        source_counts = [r['sources_count'] for r in successful_tests]
        
        print(f"\n==================== RESULTS ====================")
        print(f"✅ Successful queries: {len(successful_tests)}/{len(medical_queries)}")
        print(f"📊 Average response time: {sum(response_times)/len(response_times):.2f}s")
        print(f"📏 Average answer length: {sum(answer_lengths)/len(answer_lengths):.0f} chars")
        print(f"📚 Average sources: {sum(source_counts)/len(source_counts):.1f}")
        
        # Performance rating
        avg_time = sum(response_times) / len(response_times)
        if avg_time < 30:
            rating = "🟢 EXCELLENT"
        elif avg_time < 60:
            rating = "🟡 GOOD"
        else:
            rating = "🔴 SLOW"
        
        print(f"🎯 Performance Rating: {rating}")
    else:
        print("\n❌ No successful tests to analyze")

if __name__ == "__main__":
    test_medical_rag() 