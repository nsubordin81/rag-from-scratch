"""
Embedding models for RAG Service
Provides various embedding implementations
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import hashlib


class HuggingFaceEmbedder:
    """
    Embedder using Hugging Face Sentence Transformers models.
    Provides high-quality embeddings competitive with OpenAI's ada-002.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedder with a Hugging Face model.
        
        Args:
            model_name: Name of the Sentence Transformers model to use
                       Popular options:
                       - "all-MiniLM-L6-v2" (384 dim, fast, good quality)
                       - "all-mpnet-base-v2" (768 dim, slower, best quality)
                       - "multi-qa-MiniLM-L6-cos-v1" (384 dim, optimized for Q&A)
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a query to an embedding vector.
        
        Args:
            query: Text query to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Sentence transformers returns numpy arrays by default
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Convert multiple documents to embedding vectors.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Matrix of embedding vectors
        """
        # Batch encoding for efficiency
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a large batch of texts with explicit batch size control.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches for processing
            
        Returns:
            Matrix of embedding vectors
        """
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            batch_size=batch_size,
            show_progress_bar=True
        )
        return embeddings.astype(np.float32)


class SimpleEmbedder:
    """
    Simple embedder for testing purposes.
    Uses deterministic hash-based embeddings - not suitable for production.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the embedder.
        
        Args:
            dimension: Dimension of embeddings to generate
        """
        self.dimension = dimension
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a query to an embedding vector.
        
        Args:
            query: Text query to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Simple hash-based embedding for testing
        # In production, use proper embedding models
        hash_obj = hashlib.md5(query.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to numbers and normalize
        embedding = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            embedding.append(hash_bytes[byte_idx] / 255.0)
        
        return np.array(embedding, dtype=np.float32)
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Convert multiple documents to embedding vectors.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Matrix of embedding vectors
        """
        embeddings = []
        for doc in documents:
            embeddings.append(self.embed_query(doc))
        
        return np.array(embeddings)


# Factory function for easy embedder creation
def create_embedder(embedder_type: str = "huggingface", **kwargs):
    """
    Factory function to create embedders.
    
    Args:
        embedder_type: Type of embedder ('huggingface', 'simple')
        **kwargs: Additional arguments for the embedder
        
    Returns:
        Embedder instance
    """
    if embedder_type == "huggingface":
        return HuggingFaceEmbedder(**kwargs)
    elif embedder_type == "simple":
        return SimpleEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


# Recommended model configurations
EMBEDDING_MODELS = {
    "fast": {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast and efficient, good for most tasks"
    },
    "quality": {
        "model_name": "all-mpnet-base-v2", 
        "dimension": 768,
        "description": "Higher quality embeddings, slower"
    },
    "qa_optimized": {
        "model_name": "multi-qa-MiniLM-L6-cos-v1",
        "dimension": 384,
        "description": "Optimized for question-answering tasks"
    },
    "multilingual": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "description": "Supports multiple languages"
    }
}
