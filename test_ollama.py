#!/usr/bin/env python3
"""
Test script for Ollama LLM integration
"""

import asyncio
from rag_service.qa_service import OllamaLLM, QAService

async def test_ollama_llm():
    """Test Ollama LLM functionality"""
    
    print("🚀 Testing Ollama LLM Integration")
    
    # Test with a smaller model first (if available)
    available_models = [
        "llama3:70b-instruct",  # Your powerful model
        "codellama:code",       # Your code model
        "llama2:70b"           # Fallback
    ]
    
    ollama_llm = None
    
    # Try to connect to available models
    for model in available_models:
        try:
            print(f"\n📡 Attempting to connect to {model}...")
            ollama_llm = OllamaLLM(model_name=model, temperature=0.1)
            print(f"✅ Successfully connected to {model}")
            break
        except Exception as e:
            print(f"❌ Failed to connect to {model}: {e}")
            continue
    
    if not ollama_llm:
        print("❌ Could not connect to any Ollama models")
        return
    
    # Test basic generation
    print(f"\n🧠 Testing basic text generation...")
    try:
        response = await ollama_llm.generate("What is the capital of France? Please answer in one sentence.")
        print(f"✅ Basic generation works!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Basic generation failed: {e}")
        return
    
    # Test with QA Service
    print(f"\n🔍 Testing QA Service with Ollama...")
    qa_service = QAService(llm=ollama_llm)
    
    # Test context formatting
    context = [
        {
            "content": "France is a country in Western Europe with a rich history and culture.", 
            "metadata": {"source": "geography.txt"}
        },
        {
            "content": "Paris is the capital and largest city of France, known for the Eiffel Tower.", 
            "metadata": {"source": "cities.txt"}
        }
    ]
    
    query = "What is the capital of France and what is it known for?"
    
    try:
        result = await qa_service.generate_answer_with_sources(query, context)
        print(f"✅ QA Service with Ollama works!")
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
    except Exception as e:
        print(f"❌ QA Service failed: {e}")
        return
    
    print(f"\n🎉 Ollama integration test completed successfully!")
    print(f"Your RAG system is now powered by {ollama_llm.model_name}!")

if __name__ == "__main__":
    asyncio.run(test_ollama_llm())
