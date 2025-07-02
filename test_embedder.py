#!/usr/bin/env python3
"""
Simple test to verify HuggingFace embedder works
"""

from rag_service.embeddings import HuggingFaceEmbedder
import numpy as np

def test_embedder():
    print("Testing HuggingFace embedder...")
    
    # Create embedder
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Test query embedding
    query = "What is the capital of France?"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding type: {type(query_embedding)}")
    
    # Test document embeddings
    documents = [
        "Paris is the capital of France.",
        "Python is a programming language.",
        "Machine learning uses algorithms."
    ]
    
    doc_embeddings = embedder.embed_documents(documents)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    
    # Test similarity (simple dot product)
    similarities = np.dot(doc_embeddings, query_embedding)
    print(f"Similarities: {similarities}")
    
    # Find most similar document
    most_similar_idx = np.argmax(similarities)
    print(f"Most similar document: '{documents[most_similar_idx]}'")
    print(f"Similarity score: {similarities[most_similar_idx]:.3f}")
    
    print("Embedder test completed successfully!")

if __name__ == "__main__":
    test_embedder()
