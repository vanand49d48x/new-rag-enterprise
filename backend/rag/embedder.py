import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Text embedding will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. GPU acceleration will be disabled.")

logger = logging.getLogger(__name__)

class Embedder:
    """
    Enhanced embedder for creating text embeddings with support for multiple models.
    Supports various embedding models including sentence-transformers and custom models.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to use for computation ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name or self._get_default_model()
        self.device = device or self._get_default_device()
        self.model = None
        self.embedding_dimension = None
        
        self._initialize_model()
    
    def _get_default_model(self) -> str:
        """Get the default embedding model"""
        # Try to get from environment variable
        env_model = os.getenv('EMBEDDING_MODEL')
        if env_model:
            return env_model
        
        # Default to BGE-small-en for better English performance
        return "BAAI/bge-small-en-v1.5"
    
    def _get_default_device(self) -> str:
        """Get the default device for computation"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        elif TORCH_AVAILABLE and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not available for text embedding")
        
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            self.embedding_dimension = len(test_embedding)
            
            logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            if not text.strip():
                # Return zero vector for empty text
                return [0.0] * self.embedding_dimension
            
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            # Return zero vector on error
            return [0.0] * self.embedding_dimension
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts"""
        try:
            if not texts:
                return []
            
            # Optimized CPU settings for faster processing
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error in embedding: {e}")
            return [[0.0] * self.embedding_dimension] * len(texts)
    
    def embed_texts_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed texts in batches for better CPU performance"""
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Optimized CPU settings for faster processing
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size
            )
            
            embeddings.extend(batch_embeddings.tolist())
            
            # Log progress every 10 batches
            if batch_num % 10 == 0:
                logger.info(f"Completed {batch_num}/{total_batches} batches")
        
        logger.info(f"Completed embedding {len(texts)} texts")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.embedding_dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dimension,
            "torch_available": TORCH_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        }
    
    def change_model(self, model_name: str):
        """Change the embedding model"""
        try:
            logger.info(f"Changing embedding model from {self.model_name} to {model_name}")
            
            self.model_name = model_name
            self._initialize_model()
            
            logger.info(f"Successfully changed to model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to change model to {model_name}: {e}")
            raise
    
    def change_device(self, device: str):
        """Change the computation device"""
        try:
            logger.info(f"Changing device from {self.device} to {device}")
            
            self.device = device
            self._initialize_model()
            
            logger.info(f"Successfully changed to device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to change device to {device}: {e}")
            raise
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], candidate_embeddings: List[List[float]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with index and similarity score
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.similarity(query_embedding, candidate)
                similarities.append({
                    "index": i,
                    "similarity": similarity
                })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar embeddings: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            # Clear model from memory
            del self.model
            self.model = None
            logger.info("Embedding model cleaned up")

# Global embedder instance for standalone embed function
_global_embedder = None

def _get_global_embedder():
    """Get or create global embedder instance"""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = Embedder()
    return _global_embedder

def embed(text: str) -> List[float]:
    """
    Standalone function to embed text using the global embedder instance
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as list of floats
    """
    embedder = _get_global_embedder()
    return embedder.embed_text(text)