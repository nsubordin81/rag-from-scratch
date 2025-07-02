"""
Exceptions for RAG Service
"""

class DocumentProcessingError(Exception):
    """Exception raised for errors during document processing."""
    pass

class RetrievalError(Exception):
    """Exception raised for errors during document retrieval."""
    pass

class QAError(Exception):
    """Exception raised for errors during question answering."""
    pass

class ConfigurationError(Exception):
    """Exception raised for invalid configuration."""
    pass
