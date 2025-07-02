"""
Question-Answering Service for RAG Pipeline
Handles LLM integration and answer generation
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from .exceptions import QAError


class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass


class QAService:
    """
    Question-Answering service that uses an LLM to generate answers
    based on retrieved context documents.
    """
    
    def __init__(self, llm: BaseLLM):
        """
        Initialize QA service with an LLM.
        
        Args:
            llm: Language model instance implementing BaseLLM interface
        """
        self.llm = llm
    
    def format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context documents into a string for the LLM prompt.
        
        Args:
            context: List of context documents with 'content' and 'metadata'
            
        Returns:
            Formatted context string
        """
        if not context:
            return "No relevant context found."
        
        formatted_parts = []
        for i, doc in enumerate(context, 1):
            content = doc.get('content', '')
            source = doc.get('metadata', {}).get('source', 'Unknown source')
            formatted_parts.append(f"[{i}] {content} (Source: {source})")
        
        return "\n".join(formatted_parts)
    
    async def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate an answer to a query based on the provided context.
        
        Args:
            query: User's question
            context: List of relevant documents
            
        Returns:
            Generated answer string
            
        Raises:
            QAError: If answer generation fails
        """
        try:
            formatted_context = self.format_context(context)
            
            if context:
                prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{formatted_context}

Question: {query}

Answer:"""
            else:
                prompt = f"""Please answer the following question. Note that no specific context was provided.

Question: {query}

Answer:"""
            
            answer = await self.llm.generate(prompt)
            return answer.strip()
            
        except Exception as e:
            raise QAError(f"Failed to generate answer: {str(e)}") from e
    
    async def generate_answer_with_sources(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer with source attribution.
        
        Args:
            query: User's question
            context: List of relevant documents
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
            
        Raises:
            QAError: If answer generation fails
        """
        try:
            answer = await self.generate_answer(query, context)
            
            # Extract unique sources from context
            sources = []
            seen_sources = set()
            
            for doc in context:
                source = doc.get('metadata', {}).get('source')
                if source and source not in seen_sources:
                    sources.append(source)
                    seen_sources.add(source)
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except QAError:
            raise
        except Exception as e:
            raise QAError(f"Failed to generate answer with sources: {str(e)}") from e


# Simple LLM implementations for testing and basic usage

class MockLLM(BaseLLM):
    """Mock LLM for testing purposes"""
    
    def __init__(self, response: str = "This is a mock response."):
        self.response = response
    
    async def generate(self, prompt: str, **kwargs) -> str:
        return self.response


class SimpleLLM(BaseLLM):
    """
    Simple rule-based LLM for basic functionality.
    Not suitable for production - use proper LLM integration.
    """
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a simple response based on the prompt"""
        
        # Extract question from prompt
        if "Question:" in prompt:
            question = prompt.split("Question:")[-1].split("Answer:")[0].strip()
        else:
            question = prompt.strip()
        
        # Simple pattern matching for demo purposes
        question_lower = question.lower()
        
        if "capital" in question_lower and "france" in question_lower:
            return "Based on the provided context, the capital of France is Paris."
        elif "programming" in question_lower or "python" in question_lower:
            return "Based on the context, Python is a programming language widely used for data science."
        elif "machine learning" in question_lower or "ai" in question_lower:
            return "According to the context, machine learning is a subset of artificial intelligence that uses algorithms to learn from data."
        else:
            return "Based on the provided context, I can see relevant information but need a more specific question to provide a detailed answer."


# Factory function for easy LLM creation
def create_llm(llm_type: str = "simple", **kwargs) -> BaseLLM:
    """
    Factory function to create LLM instances.
    
    Args:
        llm_type: Type of LLM ('simple', 'mock')
        **kwargs: Additional arguments for the LLM
        
    Returns:
        LLM instance
    """
    if llm_type == "simple":
        return SimpleLLM(**kwargs)
    elif llm_type == "mock":
        return MockLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
