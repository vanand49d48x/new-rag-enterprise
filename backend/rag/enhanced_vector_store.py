"""
Enhanced Vector Store with Hybrid Retrieval
Implements enterprise-grade retrieval with BM25 + semantic search + re-ranking
"""

from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Distance, VectorParams, Filter, 
    SearchRequest, FieldCondition, MatchValue, Range
)
from qdrant_client.http.models import Batch
import uuid
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re
from backend.utils.logging_config import get_logger

from backend.rag.embedder import Embedder

logger = get_logger(__name__)

class EnterpriseVectorStore:
    """
    Enterprise-grade vector store with hybrid retrieval capabilities
    """
    
    def __init__(self, host: str = "qdrant", port: int = 6334):
        self.client = QdrantClient(host=host, port=port)
        self.embedder = Embedder()
        self.vector_size = self.embedder.get_embedding_dimension()
        self.collection_name = "enterprise_docs"
        
        # Initialize re-ranker
        try:
            self.re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logger.warning(f"Could not load re-ranker: {e}")
            self.re_ranker = None
        
        # BM25 corpus for hybrid search
        self.bm25_corpus = []
        self.bm25_model = None
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection with proper configuration"""
        try:
            # Check if collection exists by trying to get its info
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            # Collection doesn't exist or other error, try to create it
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            except Exception as create_error:
                # If creation fails (e.g., collection already exists), that's fine
                logger.info(f"Collection {self.collection_name} already exists or creation failed: {create_error}")
                pass
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Index chunks with enhanced metadata and update BM25 corpus
        """
        try:
            points = []
            bm25_documents = []
            
            for chunk in chunks:
                # Generate vector embedding
                vector = self.embedder.embed_text(chunk["text"])
                
                # Prepare payload with all metadata
                payload = {
                    "text": chunk["text"],
                    "id": chunk["id"],
                    **chunk["metadata"]
                }
                
                # Add pre-computed relevance indicators
                payload["text_length"] = len(chunk["text"])
                payload["word_count"] = len(chunk["text"].split())
                
                # Extract keywords for BM25
                keywords = self._extract_keywords(chunk["text"])
                payload["keywords"] = keywords
                
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                ))
                
                # Add to BM25 corpus
                bm25_documents.append({
                    "text": chunk["text"],
                    "keywords": keywords,
                    "metadata": chunk["metadata"]
                })
            
            # Batch upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Update BM25 model
            self._update_bm25_model(bm25_documents)
            
            logger.info(f"Indexed {len(chunks)} chunks successfully")
            
        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Return top keywords by frequency
        from collections import Counter
        keyword_freq = Counter(keywords)
        return [word for word, freq in keyword_freq.most_common(10)]
    
    def _update_bm25_model(self, documents: List[Dict[str, Any]]):
        """Update BM25 model with new documents"""
        try:
            # Prepare documents for BM25
            doc_texts = []
            for doc in documents:
                # Combine text and keywords for better BM25 performance
                combined_text = f"{doc['text']} {' '.join(doc['keywords'])}"
                doc_texts.append(combined_text.split())
            
            if doc_texts:
                self.bm25_model = BM25Okapi(doc_texts)
                self.bm25_corpus = documents
                logger.info(f"Updated BM25 model with {len(documents)} documents")
        except Exception as e:
            logger.warning(f"Could not update BM25 model: {e}")
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and BM25 retrieval
        """
        try:
            # Semantic search
            semantic_results = self._semantic_search(query, top_k * 2, filters)
            
            # BM25 search
            bm25_results = self._bm25_search(query, top_k * 2)
            
            # Combine and re-rank
            combined_results = self._combine_results(
                semantic_results, bm25_results, 
                semantic_weight, bm25_weight
            )
            
            # Re-rank with cross-encoder if available
            if self.re_ranker:
                combined_results = self._re_rank_results(query, combined_results)
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to semantic search only
            return self._semantic_search(query, top_k, filters)
    
    def _semantic_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity"""
        try:
            query_vector = self.embedder.embed_text(query)
            
            # Build filter if provided
            search_filter = None
            if filters:
                search_filter = self._build_filter(filters)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True
            )
            
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                    "search_type": "semantic"
                }
                for hit in results
            ]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 search"""
        try:
            if not self.bm25_model or not self.bm25_corpus:
                return []
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_model.get_scores(query_tokens)
            
            # Get top documents
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    doc = self.bm25_corpus[idx]
                    results.append({
                        "text": doc["text"],
                        "score": float(scores[idx]),
                        "metadata": doc["metadata"],
                        "search_type": "bm25"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _combine_results(
        self, 
        semantic_results: List[Dict], 
        bm25_results: List[Dict],
        semantic_weight: float,
        bm25_weight: float
    ) -> List[Dict]:
        """Combine semantic and BM25 results"""
        # Create lookup for quick access
        combined_lookup = {}
        
        # Add semantic results
        for result in semantic_results:
            text = result["text"]
            if text not in combined_lookup:
                combined_lookup[text] = {
                    "text": text,
                    "metadata": result["metadata"],
                    "semantic_score": result["score"],
                    "bm25_score": 0.0,
                    "combined_score": 0.0
                }
            else:
                combined_lookup[text]["semantic_score"] = result["score"]
        
        # Add BM25 results
        for result in bm25_results:
            text = result["text"]
            if text not in combined_lookup:
                combined_lookup[text] = {
                    "text": text,
                    "metadata": result["metadata"],
                    "semantic_score": 0.0,
                    "bm25_score": result["score"],
                    "combined_score": 0.0
                }
            else:
                combined_lookup[text]["bm25_score"] = result["score"]
        
        # Calculate combined scores
        for item in combined_lookup.values():
            item["combined_score"] = (
                semantic_weight * item["semantic_score"] +
                bm25_weight * item["bm25_score"]
            )
        
        # Sort by combined score
        combined_results = list(combined_lookup.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined_results
    
    def _re_rank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank results using cross-encoder"""
        try:
            if not self.re_ranker or not results:
                return results
            
            # Prepare pairs for re-ranking
            pairs = [(query, result["text"]) for result in results]
            
            # Get re-ranking scores
            scores = self.re_ranker.predict(pairs)
            
            # Update scores and re-sort
            for i, result in enumerate(results):
                result["re_rank_score"] = float(scores[i])
            
            # Sort by re-rank score
            results.sort(key=lambda x: x["re_rank_score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return results
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from filter dict"""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            elif isinstance(value, (int, float)):
                conditions.append(
                    FieldCondition(key=key, range=Range(gte=value))
                )
            elif isinstance(value, list):
                # Handle list values (e.g., file types)
                for item in value:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=item))
                    )
        
        return Filter(must=conditions) if conditions else None
    
    def search_by_metadata(self, filters: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search by metadata filters only"""
        try:
            search_filter = self._build_filter(filters)
            
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=top_k,
                with_payload=True
            )
            
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                }
                for hit in results[0]
            ]
            
        except Exception as e:
            logger.error(f"Error in metadata search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "segments_count": info.segments_count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def _enhance_query_for_enterprise_search(self, query: str) -> List[str]:
        """Enterprise-grade query enhancement for better search results"""
        enhanced_queries = [query]  # Always include original query
        
        # Extract person names and skills/keywords
        import re
        
        # Pattern to detect person + skills queries
        person_skills_pattern = r'(\w+)\'?s?\s+(skills?|abilities?|expertise|knowledge|experience|background)'
        match = re.search(person_skills_pattern, query.lower())
        
        if match:
            person_name = match.group(1)
            skill_type = match.group(2)
            
            # Create enhanced queries
            enhanced_queries.extend([
                f"{skill_type} programming languages technologies",
                f"{skill_type} computer technical",
                f"{skill_type} experience background",
                f"{person_name} resume {skill_type}",
                f"{skill_type} listed mentioned",
                f"COMPUTER SKILLS {skill_type}",  # Match document headers
                f"Programming Languages {skill_type}",
                f"technical {skill_type} expertise"
            ])
        
        # Handle general skills queries
        if 'skills' in query.lower():
            enhanced_queries.extend([
                "COMPUTER SKILLS",
                "Programming Languages",
                "technical expertise",
                "technologies experience"
            ])
        
        return list(set(enhanced_queries))  # Remove duplicates

    def enterprise_search(
        self, 
        query: str, 
        top_k: int = 10,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Enterprise-grade search with intelligent query enhancement
        """
        try:
            # Generate enhanced queries
            enhanced_queries = self._enhance_query_for_enterprise_search(query)
            
            all_results = []
            
            # Search with each enhanced query
            for enhanced_query in enhanced_queries:
                # Semantic search
                semantic_results = self._semantic_search(enhanced_query, top_k * 2, filters)
                
                # BM25 search
                bm25_results = self._bm25_search(enhanced_query, top_k * 2)
                
                # Combine results for this query
                combined_results = self._combine_results(
                    semantic_results, bm25_results, 
                    semantic_weight, bm25_weight
                )
                
                all_results.extend(combined_results)
            
            # Remove duplicates and re-rank
            unique_results = self._deduplicate_results(all_results)
            
            # Re-rank with cross-encoder if available
            if self.re_ranker:
                unique_results = self._re_rank_results(query, unique_results)
            
            return unique_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in enterprise search: {e}")
            # Fallback to original hybrid search
            return self.hybrid_search(query, top_k, semantic_weight, bm25_weight, filters)

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results while preserving best scores"""
        seen_texts = {}
        
        for result in results:
            text = result["text"]
            if text not in seen_texts:
                seen_texts[text] = result
            else:
                # Keep the result with higher score
                if result.get("combined_score", 0) > seen_texts[text].get("combined_score", 0):
                    seen_texts[text] = result
        
        # Sort by combined score
        unique_results = list(seen_texts.values())
        unique_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return unique_results 