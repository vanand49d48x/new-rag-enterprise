# backend/rag/llm.py

import os
import requests
import logging
import json
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from backend.utils.adaptive_config import get_adaptive_config

# Get adaptive config at module load
ADAPTIVE_CONFIG = get_adaptive_config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_answer(
    query: str, 
    context: str, 
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate answer using Qwen model via llama-cpp server
    """
    try:
        # Get configuration
        backend_url = os.getenv("LLM_API_URL", "http://llama-cpp:8080/completion")
        strict_mode = os.getenv("LLM_STRICT_MODE", "false").lower() == "true"
        
        if strict_mode and not context.strip():
            logger.warning("❌ Strict mode active but no context provided.")
            return "I don't have enough information to answer your question based on the available documents."
        
        # Prepare conversation history
        conversation = _prepare_conversation(query, context, history)
        
        # Prepare prompt for llama-cpp
        prompt = _build_enterprise_prompt(conversation, context)
        
        # Use adaptive config for LLM parameters
        payload = {
            "prompt": prompt,
            "n_predict": ADAPTIVE_CONFIG["max_tokens"],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stop": ["</s>", "Human:", "Assistant:", "\n\n\n"],
            "stream": False,
            "mirostat": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1
        }
        
        logger.info(f"📡 Sending request to llama-cpp server at {backend_url} with adaptive config: {ADAPTIVE_CONFIG}")
        
        # Make request to llama-cpp server
        response = requests.post(
            backend_url, 
            json=payload, 
            timeout=180,  # Further increased timeout for very large models
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("content", "").strip()
        
        if not answer:
            logger.warning("Empty response from LLM")
            return "I couldn't generate a response. Please try rephrasing your question."
        
        logger.info(f"✅ Generated answer: {len(answer)} characters")
        return answer
        
    except requests.exceptions.Timeout as e:
        logger.error(f"❌ LLM server timeout: {e}")
        return "⚠️ The language model is taking too long to respond. This can happen with large models. Please try a simpler question or wait a moment and try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ LLM server error: {e}")
        return "⚠️ I'm having trouble connecting to the language model. Please try again later."
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        return "⚠️ An error occurred while generating the response. Please try again."

async def generate_answer_stream(
    query: str, 
    context: str, 
    history: Optional[List[Dict[str, str]]] = None
) -> AsyncGenerator[str, None]:
    """
    Generate streaming answer using Qwen model via llama-cpp server
    """
    try:
        # Get configuration
        backend_url = os.getenv("LLM_API_URL", "http://llama-cpp:8080/completion")
        strict_mode = os.getenv("LLM_STRICT_MODE", "false").lower() == "true"
        
        if strict_mode and not context.strip():
            logger.warning("❌ Strict mode active but no context provided.")
            yield "I don't have enough information to answer your question based on the available documents."
            return
        
        # Prepare conversation history
        conversation = _prepare_conversation(query, context, history)
        
        # Prepare prompt for llama-cpp
        prompt = _build_enterprise_prompt(conversation, context)
        
        # Prepare payload for llama-cpp server with streaming enabled
        payload = {
            "prompt": prompt,
            "n_predict": 64,  # Reduced for faster response
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stop": ["</s>", "Human:", "Assistant:", "\n\n\n"],
            "stream": True,  # Enable streaming
            "mirostat": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1
        }
        
        logger.info(f"📡 Sending streaming request to llama-cpp server at {backend_url}")
        logger.info(f"📝 Prompt length: {len(prompt)} characters")
        
        # Make streaming request to llama-cpp server
        response = requests.post(
            backend_url, 
            json=payload, 
            timeout=300,  # Increased timeout for streaming
            headers={"Content-Type": "application/json"},
            stream=True
        )
        response.raise_for_status()
        logger.info("✅ Received streaming response from llama-cpp server")
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                try:
                    line_str = line.decode('utf-8')
                    # Handle data: prefix from llama-cpp server
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]  # Remove 'data: ' prefix
                    
                    data = json.loads(line_str)
                    if 'content' in data and data['content']:
                        token = data['content']
                        yield token
                        # Stop if we get a stop signal
                        if data.get('stop', False):
                            break
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing streaming token: {e}")
                    continue
        
        logger.info("✅ Streaming answer generation completed")
        
    except requests.exceptions.Timeout as e:
        logger.error(f"❌ LLM server timeout: {e}")
        yield "⚠️ The language model is taking too long to respond. This can happen with large models. Please try a simpler question or wait a moment and try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ LLM server error: {e}")
        yield "⚠️ I'm having trouble connecting to the language model. Please try again later."
    except Exception as e:
        logger.error(f"❌ Error generating streaming answer: {e}")
        yield "⚠️ An error occurred while generating the response. Please try again."

def _prepare_conversation(
    query: str, 
    context: str, 
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Prepare conversation history for the LLM"""
    conversation = []
    
    # Add system message
    conversation.append({
        "role": "system",
        "content": "You are an enterprise AI assistant with access to a comprehensive knowledge base. Your role is to provide accurate, helpful answers based on the provided context. Always cite your sources and be transparent about what information you have available."
    })
    
    # Add conversation history if available
    if history:
        for msg in history[-5:]:  # Limit to last 5 messages
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                conversation.append(msg)
    
    # Add current context and query
    conversation.append({
        "role": "user",
        "content": f"Based on the following context, please answer my question:\n\nContext:\n{context}\n\nQuestion: {query}"
    })
    
    return conversation

def _build_enterprise_prompt(conversation: List[Dict[str, str]], context: str) -> str:
    """Build enterprise-grade prompt for the LLM"""
    
    # Start with system instructions
    prompt = """<|im_start|>system
You are an enterprise AI assistant with access to a comprehensive knowledge base. Your role is to provide accurate, helpful answers based on the provided context.

IMPORTANT GUIDELINES:
1. Base your answers ONLY on the provided context below
2. If the context doesn't contain enough information, say so clearly
3. Cite specific sources when possible
4. Be concise but comprehensive
5. Use professional, business-appropriate language
6. If you're unsure about something, acknowledge the uncertainty
7. Provide actionable insights when possible
8. ALWAYS use the information provided in the context - do not say you cannot access documents
9. When asked about a person's skills, connect the person to their skills from the documents
10. If you find skills information in the context, present it clearly and comprehensively
11. Look for patterns like "COMPUTER SKILLS", "Programming Languages", etc. in the context
12. When someone asks about "X's skills", search the context for skills information related to X

Context Information:
{context}

<|im_end|>
"""
    
    # Add conversation history
    for msg in conversation[1:]:  # Skip system message as it's already added
        if msg["role"] == "user":
            prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    
    # Add final instruction
    prompt += "<|im_start|>assistant\n"
    
    return prompt

def generate_summary(text: str, max_length: int = 200) -> str:
    """Generate a summary of the provided text"""
    try:
        backend_url = os.getenv("LLM_API_URL", "http://llama-cpp:8080/completion")
        
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Please provide a concise summary of the following text in {max_length} characters or less.

<|im_end|>
<|im_start|>user
Please summarize this text:

{text}

<|im_end|>
<|im_start|>assistant
"""
        
        payload = {
            "prompt": prompt,
            "n_predict": 100,
            "temperature": 0.3,
            "top_p": 0.9,
            "stop": ["</s>", "Human:", "Assistant:", "\n\n"]
        }
        
        response = requests.post(
            backend_url, 
            json=payload, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        summary = result.get("content", "").strip()
        
        return summary if summary else text[:max_length] + "..."
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return text[:max_length] + "..."

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using LLM"""
    try:
        backend_url = os.getenv("LLM_API_URL", "http://llama-cpp:8080/completion")
        
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Extract the {max_keywords} most important keywords from the following text. Return only the keywords separated by commas, no explanations.

<|im_end|>
<|im_start|>user
Extract keywords from this text:

{text}

<|im_end|>
<|im_start|>assistant
"""
        
        payload = {
            "prompt": prompt,
            "n_predict": 50,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["</s>", "Human:", "Assistant:", "\n"]
        }
        
        response = requests.post(
            backend_url, 
            json=payload, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        keywords_text = result.get("content", "").strip()
        
        # Parse keywords
        keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
        return keywords[:max_keywords]
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []
