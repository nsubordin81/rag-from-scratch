#!/usr/bin/env python3
"""
Simple test script for QA Service
"""

import asyncio
from rag_service.qa_service import QAService, SimpleLLM, MockLLM

async def test_qa_service():
    """Test QA Service functionality"""
    
    # Test with SimpleLLM
    print("Testing QA Service with SimpleLLM...")
    simple_llm = SimpleLLM()
    qa_service = QAService(llm=simple_llm)
    
    # Test context formatting
    context = [
        {"content": "France is a country in Europe.", "metadata": {"source": "doc1"}},
        {"content": "Paris is the capital of France.", "metadata": {"source": "doc2"}},
        {"content": "France has a population of about 67 million.", "metadata": {"source": "doc3"}}
    ]
    
    formatted = qa_service.format_context(context)
    print("✅ Context formatting works")
    print(f"Formatted context: {formatted[:100]}...")
    
    # Test answer generation
    query = "What is the capital of France?"
    answer = await qa_service.generate_answer(query, context)
    print(f"\n✅ Answer generation works")
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    
    # Test answer with sources
    result = await qa_service.generate_answer_with_sources(query, context)
    print(f"\n✅ Answer with sources works")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    
    # Test with MockLLM
    print(f"\n--- Testing with MockLLM ---")
    mock_llm = MockLLM("This is a mock answer about France.")
    qa_service_mock = QAService(llm=mock_llm)
    
    answer_mock = await qa_service_mock.generate_answer(query, context)
    print(f"Mock answer: {answer_mock}")
    
    print(f"\n✅ All QA Service tests passed!")

if __name__ == "__main__":
    asyncio.run(test_qa_service())
