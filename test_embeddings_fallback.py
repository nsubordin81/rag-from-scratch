#!/usr/bin/env python3
"""
Test script that tries HuggingFace embedder with fallback to SimpleEmbedder
"""

import numpy as np
from rag_service.document_processor import DocumentProcessor
from rag_service.vector_store import VectorStore
from rag_service.retriever import Retriever

def test_with_fallback():
    """Test retrieval with fallback embedding strategy"""
    
    # Try to use HuggingFace embedder, fallback to SimpleEmbedder
    try:
        from rag_service.embeddings import HuggingFaceEmbedder
        print("Attempting to load HuggingFace embedder...")
        embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
        print(f"✅ HuggingFace embedder loaded successfully. Dimension: {embedder.dimension}")
        dimension = embedder.dimension
    except Exception as e:
        print(f"❌ Failed to load HuggingFace embedder: {e}")
        print("🔄 Falling back to SimpleEmbedder...")
        from rag_service.embeddings import SimpleEmbedder
        embedder = SimpleEmbedder(dimension=384)
        dimension = 384
        print(f"✅ SimpleEmbedder loaded. Dimension: {dimension}")
    
    # Create components
    vector_store = VectorStore(dimension=dimension)
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
    print("Generating embeddings for documents...")
    doc_texts = [doc["content"] for doc in documents]
    embeddings = embedder.embed_documents(doc_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Add documents to vector store
    vector_store.add_documents(documents, embeddings)
    print(f"Added {vector_store.count()} documents to vector store")
    
    # Test retrieval
    query = "What is the capital of France?"
    print(f"\nQuery: '{query}'")
    results = retriever.retrieve(query, k=3)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        print(f"   Source: {result['metadata']['source']}")
    
    # Test that the geography document has the highest score (if using good embeddings)
    if len(results) > 0:
        top_result = results[0]
        expected_source = "geography.txt"
        if top_result['metadata']['source'] == expected_source:
            print(f"\n✅ Success! Top result is from {expected_source} as expected")
        else:
            print(f"\n⚠️  Top result is from {top_result['metadata']['source']}, expected {expected_source}")
    
    # Clean up
    vector_store.cleanup()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_with_fallback()
