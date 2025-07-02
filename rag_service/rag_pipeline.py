"""
RAG Pipeline - Main orchestrator for the Retrieval-Augmented Generation system
Coordinates document ingestion and query processing
"""

import os
from typing import List, Dict, Any, Optional, Union
import asyncio
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .retriever import Retriever
from .qa_service import QAService, SimpleLLM
from .embeddings import HuggingFaceEmbedder, SimpleEmbedder
from .exceptions import DocumentProcessingError, RetrievalError, QAError, ConfigurationError


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates document ingestion and query processing.
    Integrates all components: DocumentProcessor, VectorStore, Retriever, and QAService.
    """
    
    def __init__(self, 
                 document_processor: Optional[DocumentProcessor] = None,
                 vector_store: Optional[VectorStore] = None,
                 retriever: Optional[Retriever] = None,
                 qa_service: Optional[QAService] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG pipeline with components.
        
        Args:
            document_processor: Document processing component
            vector_store: Vector storage component
            retriever: Document retrieval component
            qa_service: Question-answering component
            config: Configuration dictionary for components
        """
        self.config = config or {}
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components with defaults if not provided
        if document_processor is None:
            self.document_processor = DocumentProcessor()
        else:
            self.document_processor = document_processor
            
        if vector_store is None:
            # Default dimension for embeddings
            dimension = self.config.get('vector_store', {}).get('dimension', 384)
            self.vector_store = VectorStore(dimension=dimension)
        else:
            self.vector_store = vector_store
            
        if retriever is None:
            # Create default embedder
            try:
                embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
            except Exception:
                # Fallback to SimpleEmbedder
                embedder = SimpleEmbedder(dimension=384)
            
            self.retriever = Retriever(
                vector_store=self.vector_store,
                embedder=embedder
            )
        else:
            self.retriever = retriever
            
        if qa_service is None:
            # Create default LLM and QA service
            llm = SimpleLLM()
            self.qa_service = QAService(llm=llm)
        else:
            self.qa_service = qa_service
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if not self.config:
            return
        
        # Validate vector store config
        if 'vector_store' in self.config:
            vs_config = self.config['vector_store']
            if 'dimension' in vs_config:
                if vs_config['dimension'] <= 0:
                    raise ConfigurationError("Vector store dimension must be positive")
        
        # Validate retriever config
        if 'retriever' in self.config:
            ret_config = self.config['retriever']
            if 'k' in ret_config:
                if ret_config['k'] <= 0:
                    raise ConfigurationError("Retriever k must be positive")
            if 'threshold' in ret_config:
                if not (0.0 <= ret_config['threshold'] <= 1.0):
                    raise ConfigurationError("Retriever threshold must be between 0 and 1")
        
        # Validate QA service config
        if 'qa_service' in self.config:
            qa_config = self.config['qa_service']
            if 'temperature' in qa_config:
                if not (0.0 <= qa_config['temperature'] <= 1.0):
                    raise ConfigurationError("QA service temperature must be between 0 and 1")
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self._validate_config()
            return True
        except ConfigurationError:
            raise

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document into the RAG system.
        
        Args:
            file_path: Path to the document to ingest
            
        Returns:
            Dictionary with ingestion results
            
        Raises:
            DocumentProcessingError: If document processing fails
        """
        try:
            # Process the document
            doc_data = self.document_processor.process_file(file_path)
            
            # Chunk the content
            chunks = self.document_processor.chunk_text(
                doc_data["content"],
                chunk_size=1000,
                overlap=200
            )
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_id = f"{file_path}_chunk_{i}"
                documents.append({
                    "id": doc_id,
                    "content": chunk,
                    "metadata": {
                        **doc_data["metadata"],
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
            
            # Generate embeddings
            doc_texts = [doc["content"] for doc in documents]
            embeddings = self.retriever.embedder.embed_documents(doc_texts)
            
            # Add to vector store
            self.vector_store.add_documents(documents, embeddings)
            
            return {
                "file_path": file_path,
                "chunks_created": len(chunks),
                "documents_added": len(documents),
                "status": "success"
            }
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to ingest document {file_path}: {str(e)}") from e
    
    def ingest_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents into the RAG system.
        
        Args:
            file_paths: List of paths to documents to ingest
            
        Returns:
            List of ingestion results for each document
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.ingest_document(file_path)
                results.append(result)
            except DocumentProcessingError as e:
                results.append({
                    "file_path": file_path,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    async def query(self, 
                    query: str, 
                    k: int = 5, 
                    threshold: Optional[float] = None,
                    metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            threshold: Minimum similarity threshold
            metadata_filter: Filter for document metadata
            
        Returns:
            Dictionary with answer and sources
            
        Raises:
            RetrievalError: If document retrieval fails
            QAError: If answer generation fails
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(
                query,
                k=k,
                threshold=threshold,
                metadata_filter=metadata_filter
            )
            
            # Generate answer with sources
            result = await self.qa_service.generate_answer_with_sources(
                query,
                retrieved_docs
            )
            
            # Add retrieval metadata
            result.update({
                "query": query,
                "retrieved_documents": len(retrieved_docs),
                "retrieval_scores": [doc.get("score", 0.0) for doc in retrieved_docs]
            })
            
            return result
            
        except (RetrievalError, QAError):
            raise
        except Exception as e:
            raise QAError(f"Failed to process query: {str(e)}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "total_documents": self.vector_store.count(),
            "vector_store_dimension": getattr(self.vector_store, 'dimension', 'unknown'),
            "embedder_type": type(self.retriever.embedder).__name__,
            "llm_type": type(self.qa_service.llm).__name__
        }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.vector_store, 'cleanup'):
            self.vector_store.cleanup()
    
    @classmethod
    def from_environment(cls, **kwargs):
        """
        Create RAG pipeline from environment configuration.
        This is a placeholder for environment-based initialization.
        """
        # This would typically read from environment variables
        # For now, return a default instance
        return cls(**kwargs)


# Convenience function for quick setup
def create_rag_pipeline(config: Optional[Dict[str, Any]] = None) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with sensible defaults.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RAG pipeline instance
    """
    return RAGPipeline(config=config)
