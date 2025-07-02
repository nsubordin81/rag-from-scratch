#!/usr/bin/env python3
"""
Simple test script to verify our Retriever works with real components
"""

import numpy as np
from rag_service.document_processor import DocumentProcessor
from rag_service.vector_store import VectorStore
from rag_service.retriever import Retriever, SimpleEmbedder
import tempfile

def test_end_to_end_retrieval():
    """Test the retrieval system with real components"""
    
    # Create components
    processor = DocumentProcessor()
    vector_store = VectorStore(dimension=384)
    embedder = SimpleEmbedder(dimension=384)
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    
    # Create sample documents
    documents = [
        {
            "id": "doc1",
            "content": "The capital of France is Paris. It is a beautiful city with many museums.",
            "metadata": {"source": "geography.txt", "topic": "geography"}
        },
        {
            "id": "doc2", 
            "content": "Python is a programming language. It is widely used for data science.",
            "metadata": {"source": "programming.txt", "topic": "programming"}
        },
        {
            "id": "doc3",
            "content": "Machine learning is a subset of artificial intelligence. It uses algorithms to learn from data.",
            "metadata": {"source": "ai.txt", "topic": "AI"}
        }
    ]
    
    # Generate embeddings for documents
    doc_texts = [doc["content"] for doc in documents]
    embeddings = embedder.embed_documents(doc_texts)
    
    # Add documents to vector store
    vector_store.add_documents(documents, embeddings)
    
    print(f"Added {vector_store.count()} documents to vector store")
    
    # Test retrieval
    query = "What is the capital of France?"
    results = retriever.retrieve(query, k=2)
    
    print(f"\nQuery: '{query}'")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        print(f"   Source: {result['metadata']['source']}")
    
    # Test with threshold
    print(f"\n--- Testing with threshold ---")
    filtered_results = retriever.retrieve(query, k=5, threshold=0.5)
    print(f"Results above threshold 0.5: {len(filtered_results)}")
    
    # Test with metadata filter
    print(f"\n--- Testing with metadata filter ---")
    geo_results = retriever.retrieve(query, k=5, metadata_filter={"topic": "geography"})
    print(f"Results with topic='geography': {len(geo_results)}")
    
    # Clean up
    vector_store.cleanup()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_end_to_end_retrieval()
