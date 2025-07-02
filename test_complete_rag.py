#!/usr/bin/env python3
"""
End-to-end test of the complete RAG pipeline
"""

import asyncio
import tempfile
import os
from rag_service.rag_pipeline import RAGPipeline

async def test_complete_rag_pipeline():
    """Test the complete RAG pipeline end-to-end"""
    
    print("🚀 Testing Complete RAG Pipeline")
    
    # Create a RAG pipeline with default components
    print("Creating RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Display pipeline stats
    stats = pipeline.get_stats()
    print(f"Pipeline initialized with:")
    print(f"  - Embedder: {stats['embedder_type']}")
    print(f"  - LLM: {stats['llm_type']}")
    print(f"  - Vector Store Dimension: {stats['vector_store_dimension']}")
    
    # Create sample documents for ingestion
    print("\n📄 Creating sample documents...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample text files
        doc1_path = os.path.join(temp_dir, "geography.txt")
        with open(doc1_path, 'w') as f:
            f.write("France is a country in Western Europe. The capital of France is Paris, which is known for its art, culture, and the Eiffel Tower. France has a population of approximately 67 million people.")
        
        doc2_path = os.path.join(temp_dir, "programming.txt") 
        with open(doc2_path, 'w') as f:
            f.write("Python is a high-level programming language known for its simplicity and readability. It is widely used in data science, web development, and artificial intelligence applications.")
        
        doc3_path = os.path.join(temp_dir, "science.txt")
        with open(doc3_path, 'w') as f:
            f.write("Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.")
        
        # Test document ingestion
        print("\n📥 Ingesting documents...")
        
        ingestion_results = pipeline.ingest_documents([doc1_path, doc2_path, doc3_path])
        
        for result in ingestion_results:
            if result['status'] == 'success':
                print(f"✅ {os.path.basename(result['file_path'])}: {result['chunks_created']} chunks, {result['documents_added']} documents")
            else:
                print(f"❌ {os.path.basename(result['file_path'])}: {result['error']}")
        
        # Check pipeline stats after ingestion
        stats = pipeline.get_stats()
        print(f"\nTotal documents in system: {stats['total_documents']}")
        
        # Test queries
        print("\n🔍 Testing queries...")
        
        queries = [
            "What is the capital of France?",
            "What is Python used for?", 
            "What is machine learning?"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            
            try:
                result = await pipeline.query(query, k=3)
                
                print(f"Answer: {result['answer']}")
                print(f"Sources: {', '.join(result['sources'])}")
                print(f"Retrieved {result['retrieved_documents']} documents")
                
                if result['retrieval_scores']:
                    print(f"Top score: {max(result['retrieval_scores']):.3f}")
                    
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        # Test query with custom parameters
        print(f"\n🎯 Testing query with custom parameters...")
        try:
            result = await pipeline.query(
                "Tell me about France", 
                k=2, 
                threshold=0.1
            )
            print(f"Custom query result: {result['answer'][:100]}...")
            print(f"Retrieved {result['retrieved_documents']} documents with threshold 0.1")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Clean up
    pipeline.cleanup()
    print(f"\n✅ RAG Pipeline test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_complete_rag_pipeline())
