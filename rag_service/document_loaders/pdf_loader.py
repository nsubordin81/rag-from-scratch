"""
PDF loader for RAG service
"""
class PyPDFLoader:
    """
    Mock PyPDFLoader for testing purposes.
    In a real implementation, this would use PyPDF2 or similar to load PDF files.
    """
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        """
        Mock load method that returns dummy document data.
        In a real implementation, this would parse the PDF.
        """
        return [
            MockDocument(page_content="Page 1 content", metadata={"page": 1}),
            MockDocument(page_content="Page 2 content", metadata={"page": 2})
        ]

class MockDocument:
    """Mock document class used by PyPDFLoader"""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
