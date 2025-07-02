#!/usr/bin/env python3
"""
Simple test for QA Service
"""

import asyncio
from unittest.mock import Mock
from rag_service.qa_service import QAService

async def test_qa_service():
    """Simple test of QA Service functionality"""
    
    # Create a mock LLM
    mock_llm = Mock()
    mock_llm.generate = Mock(return_value="Paris is the capital of France.")
    
    # Create QA service
    qa_service = QAService(llm=mock_llm)
    
    # Test context formatting
    context = [
        {"content": "Paris is the capital of France", "metadata": {"source": "geo.txt"}},
        {"content": "France is in Europe", "metadata": {"source": "geo.txt"}}
    ]
    formatted = qa_service.format_context(context)
    print(f"Formatted context:\n{formatted}")
    
    # Test answer generation
    query = "What is the capital of France?"
    answer = await qa_service.generate_answer(query, context)
    print(f"\nAnswer: {answer}")
    
    # Test answer with sources
    answer_with_sources = await qa_service.generate_answer_with_sources(query, context)
    print(f"\nAnswer with sources: {answer_with_sources}")
    
    print("\n✅ QA Service test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_qa_service())
