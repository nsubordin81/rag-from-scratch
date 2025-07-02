"""
Retriever for RAG Service
Handles document retrieval based on query similarity
"""

import numpy as np
from typing import List, Dict, Any, Optional

from rag_service.exceptions import RetrievalError


class Retriever:
    """
    Document retriever that uses embeddings to find relevant documents.
    Combines an embedder and vector store to retrieve documents based on query similarity.
    """
    
    def __init__(self, vector_store=None, embedder=None):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance for similarity search
            embedder: Embedder instance for converting queries to embeddings
        """
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = 5, threshold: float = 0.0, 
                metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: Text query to search for
            k: Number of documents to retrieve
            threshold: Minimum similarity score threshold
            metadata_filter: Optional metadata filter (not implemented in this basic version)
            
        Returns:
            List of relevant documents with similarity scores
        """
        if not self.embedder:
            raise RetrievalError("No embedder configured for retrieval")
        
        if not self.vector_store:
            raise RetrievalError("No vector store configured for retrieval")
        
        try:
            # Convert query to embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Ensure it's a numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            # Search for similar documents
            results = self.vector_store.similarity_search(query_embedding, k=k)
            
            # Apply threshold filtering
            if threshold > 0.0:
                results = [result for result in results if result["score"] >= threshold]
            
            # Apply metadata filtering (basic implementation)
            if metadata_filter:
                filtered_results = []
                for result in results:
                    doc_metadata = result["document"].get("metadata", {})
                    # Check if all filter criteria match
                    if all(doc_metadata.get(key) == value for key, value in metadata_filter.items()):
                        filtered_results.append(result)
                results = filtered_results
            
            # Extract just the content and metadata for the return format
            formatted_results = []
            for result in results:
                document = result["document"]
                formatted_results.append({
                    "content": document["content"],
                    "metadata": document["metadata"],
                    "score": result["score"]
                })
            
            return formatted_results
            
        except Exception as e:
            raise RetrievalError(f"Error during document retrieval: {str(e)}")
