"""
Document Processor for RAG Service
Handles the ingestion and processing of documents
"""

import os
from pathlib import Path
from datetime import datetime
import mimetypes
from typing import List, Dict, Any, Optional

from rag_service.exceptions import DocumentProcessingError
from rag_service.document_loaders.pdf_loader import PyPDFLoader


class DocumentProcessor:
    """
    Processes documents for a RAG application.
    Handles reading files, extracting content, and preparing them for embedding.
    """
    
    def __init__(self):
        # Initialize supported file types
        self.supported_file_types = {'.txt', '.pdf'}
        
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document file and extract its content and metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dict containing processed content and metadata
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        # Check file type
        if path.suffix.lower() not in self.supported_file_types:
            raise DocumentProcessingError(f"Unsupported file type: {path.suffix}")
        
        # Extract metadata
        metadata = self.extract_metadata(file_path)            # Read and process based on file type
        try:
            if path.suffix.lower() == '.txt':
                content = self._process_text_file(file_path)
            elif path.suffix.lower() == '.pdf':
                content = self._process_pdf_file(file_path)
            else:
                # This shouldn't happen due to the check above, but just in case
                raise DocumentProcessingError(f"No processor available for {path.suffix}")
                
            # Check for empty content
            if not content or content.strip() == "":
                raise DocumentProcessingError(f"Empty content in file: {file_path}")
                
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            raise DocumentProcessingError(f"Error processing file {file_path}: {str(e)}")
    
    def _process_text_file(self, file_path: str) -> str:
        """Process a plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _process_pdf_file(self, file_path: str) -> str:
        """Process a PDF file using PyPDFLoader"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Combine all page contents
        content = "\n\n".join([doc.page_content for doc in documents])
        return content
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing metadata fields
        """
        path = Path(file_path)
        stats = path.stat()
        
        return {
            "filename": path.name,
            "file_size": stats.st_size,
            "file_type": path.suffix.lower(),
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        }
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks of specified size.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Simple character-based chunking
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position with overlap
            end = min(start + chunk_size, text_length)
            
            # Extract chunk
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move to next chunk position, accounting for overlap
            start = end - overlap if end < text_length else text_length
            
        return chunks
    
    def get_processor_type(self, file_path: str) -> str:
        """
        Determine the processor type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String identifier for the processor type
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == '.txt':
            return "TextProcessor"
        elif ext == '.pdf':
            return "PDFProcessor"
        elif ext == '.docx':
            return "DocxProcessor"
        elif ext == '.md':
            return "MarkdownProcessor"
        else:
            raise DocumentProcessingError(f"Unsupported file extension: {ext}")
