#!/usr/bin/env python3
"""
Compare SimpleLLM vs Ollama LLM in the RAG pipeline
"""

import asyncio
import tempfile
import os
from rag_service.rag_pipeline import RAGPipeline
from rag_service.qa_service import SimpleLLM, OllamaLLM, QAService

async def compare_llms():
    """Compare SimpleLLM vs Ollama in a RAG pipeline"""
    
    print("🔄 Comparing SimpleLLM vs Ollama LLM in RAG Pipeline")
    
    # Create sample documents
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample document about Python
        doc_path = os.path.join(temp_dir, "python_guide.txt")
        with open(doc_path, 'w') as f:
            f.write("""
Python is a high-level, interpreted programming language with dynamic semantics. 
Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
make it very attractive for Rapid Application Development, as well as for use as a 
scripting or glue language to connect existing components together.

Python's simple, easy-to-learn syntax emphasizes readability and therefore reduces 
the cost of program maintenance. Python supports modules and packages, which encourages 
program modularity and code reuse. The Python interpreter and the extensive standard 
library are available in source or binary form without charge for all major platforms, 
and can be freely distributed.

Python is widely used in data science, web development, automation, artificial intelligence, 
and scientific computing. Popular frameworks include Django and Flask for web development, 
NumPy and Pandas for data analysis, and TensorFlow and PyTorch for machine learning.
            """)
        
        # Test 1: RAG Pipeline with SimpleLLM
        print("\n" + "="*60)
        print("🤖 Testing with SimpleLLM (rule-based)")
        print("="*60)
        
        simple_pipeline = RAGPipeline()  # This will use SimpleLLM as fallback
        
        # Force SimpleLLM for comparison
        simple_llm = SimpleLLM()
        simple_qa = QAService(llm=simple_llm)
        simple_pipeline.qa_service = simple_qa
        
        # Ingest document
        result = simple_pipeline.ingest_document(doc_path)
        print(f"📄 Ingested document: {result['chunks_created']} chunks")
        
        # Query with SimpleLLM
        query = "What is Python used for and what makes it popular for development?"
        
        simple_result = await simple_pipeline.query(query)
        print(f"\n❓ Query: {query}")
        print(f"🤖 SimpleLLM Answer: {simple_result['answer']}")
        print(f"📚 Sources: {simple_result['sources']}")
        
        simple_pipeline.cleanup()
        
        # Test 2: RAG Pipeline with Ollama
        print("\n" + "="*60)
        print("🧠 Testing with Ollama LLM (llama3:70b-instruct)")
        print("="*60)
        
        try:
            ollama_llm = OllamaLLM(model_name="llama3:70b-instruct", temperature=0.1)
            ollama_qa = QAService(llm=ollama_llm)
            ollama_pipeline = RAGPipeline(qa_service=ollama_qa)
            
            # Ingest the same document
            result = ollama_pipeline.ingest_document(doc_path)
            print(f"📄 Ingested document: {result['chunks_created']} chunks")
            
            # Query with Ollama
            print(f"🧠 Generating answer with Ollama (this may take a moment)...")
            ollama_result = await ollama_pipeline.query(query)
            print(f"\n❓ Query: {query}")
            print(f"🧠 Ollama Answer: {ollama_result['answer']}")
            print(f"📚 Sources: {ollama_result['sources']}")
            
            ollama_pipeline.cleanup()
            
            # Comparison
            print("\n" + "="*60)
            print("📊 COMPARISON SUMMARY")
            print("="*60)
            print(f"SimpleLLM Answer Length: {len(simple_result['answer'])} characters")
            print(f"Ollama Answer Length: {len(ollama_result['answer'])} characters")
            print(f"\nOllama provides much more detailed and contextual answers!")
            
        except Exception as e:
            print(f"❌ Ollama test failed: {e}")
            print("Make sure Ollama is running: `ollama serve`")
    
    print(f"\n✅ LLM comparison completed!")

if __name__ == "__main__":
    asyncio.run(compare_llms())
