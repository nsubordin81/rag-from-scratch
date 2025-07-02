"""
Vector Store for RAG Service
Implements vector storage and similarity search using Chromadb
"""

import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

from rag_service.exceptions import RetrievalError


class VectorStore:
    """
    Vector store for storing and retrieving document embeddings.
    Uses Chromadb as the backend for efficient similarity search.
    """
    
    def __init__(self, dimension: int = 384, persist_directory: str = None, collection_name: str = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of embeddings
            persist_directory: Directory to persist Chroma data (None for in-memory)
            collection_name: Name of the collection (defaults to a unique name)
        """
        self.dimension = dimension
        self.persist_directory = persist_directory
        
        # Create Chroma client (in-memory by default)
        self.client = chromadb.Client()
        
        # Use a unique collection name for each instance to avoid conflicts in tests
        if collection_name is None:
            collection_name = f"documents_{uuid.uuid4().hex}"
            
        # Create collection for storing documents and embeddings
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"dimension": dimension},
        )
        
        # Document ID to document mapping for quick lookup
        self.documents = {}
        
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Add documents with their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries with 'id', 'content', and 'metadata'
            embeddings: Document embeddings as a numpy array (n_docs, dimension)
        """
        if len(documents) == 0:
            return
            
        if len(documents) != embeddings.shape[0]:
            raise ValueError(f"Number of documents ({len(documents)}) doesn't match number of embeddings ({embeddings.shape[0]})")
        
        # Extract necessary data for Chroma
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Convert embeddings to list format for Chroma
        embeddings_list = embeddings.tolist()
        
        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=contents,
            metadatas=metadatas
        )
        
        # Update our document cache
        for doc in documents:
            self.documents[doc["id"]] = doc
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of results with 'document' and 'score' keys
        """
        if self.count() == 0:
            return []
            
        # Convert query embedding to list for Chroma
        query_embedding_list = query_embedding.tolist()
        
        # Search in Chroma
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # No results
            if len(results["ids"]) == 0 or len(results["ids"][0]) == 0:
                return []
                
            # Format results to match expected output
            formatted_results = []
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                distance = results["distances"][0][i]
                
                # Chroma returns distances, convert to similarity score (1 - distance)
                # This assumes L2 distance, adjust if using cosine or dot product
                score = 1.0 - min(1.0, distance)
                
                # Get the full document from our cache
                document = self.get_document(doc_id)
                
                formatted_results.append({
                    "document": document,
                    "score": score
                })
                
            return formatted_results
            
        except Exception as e:
            raise RetrievalError(f"Error during similarity search: {str(e)}")
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary or None if not found
        """
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID
        """
        try:
            self.collection.delete(ids=[doc_id])
            
            # Remove from document cache
            if doc_id in self.documents:
                del self.documents[doc_id]
                
        except Exception as e:
            raise RetrievalError(f"Error deleting document {doc_id}: {str(e)}")
    
    def update_document(self, document: Dict[str, Any], embedding: np.ndarray) -> None:
        """
        Update a document and its embedding.
        
        Args:
            document: Updated document dictionary
            embedding: Updated embedding vector
        """
        doc_id = document["id"]
        
        # Delete existing document
        self.delete_document(doc_id)
        
        # Add updated document
        self.add_documents([document], np.array([embedding]))
    
    def count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Document count
        """
        return self.collection.count()
    
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        """
        # Get all document IDs
        if self.count() > 0:
            # Get the current collection name
            collection_name = self.collection.name
            
            # Clear the collection
            self.collection.delete(where={})
            
            # Clear document cache
            self.documents = {}
            
    def cleanup(self) -> None:
        """
        Clean up resources used by the vector store.
        Deletes the collection from the client to prevent buildup of collections.
        """
        try:
            # Only try to delete if we have a reference to the collection
            if hasattr(self, 'collection') and self.collection is not None:
                collection_name = self.collection.name
                
                # Check if collection exists before trying to delete it
                try:
                    self.client.get_collection(collection_name)
                    self.client.delete_collection(collection_name)
                except Exception:
                    # Collection doesn't exist or already deleted
                    pass
                
                # Clear our reference to prevent double-deletion
                self.collection = None
        except Exception as e:
            # Log but don't raise - this is cleanup code
            print(f"Warning: Error during vector store cleanup: {str(e)}")
    
    def __del__(self):
        """
        Destructor to ensure cleanup when the instance is garbage collected.
        """
        self.cleanup()
