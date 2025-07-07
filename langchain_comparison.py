#!/usr/bin/env python3
"""
Example of the same RAG system using LangChain
This would replace most of our custom implementation
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pathlib import Path
import tempfile
import asyncio

class LangChainRAGPipeline:
    """
    LangChain-based RAG pipeline - much more concise!
    """
    
    def __init__(self, model_name: str = "llama3:70b-instruct"):
        # Initialize components with LangChain
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.llm = Ollama(model=model_name, temperature=0.1)
        
        # Create vector store (persistent)
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
    
    def ingest_document(self, file_path: str):
        """Ingest a document into the vector store"""
        # Load document
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        docs = loader.load()
        
        # Split into chunks
        splits = self.text_splitter.split_documents(docs)
        
        # Add to vector store
        self.vectorstore.add_documents(splits)
        
        return {"chunks_created": len(splits), "status": "success"}
    
    def ingest_documents(self, file_paths: list):
        """Ingest multiple documents"""
        results = []
        for path in file_paths:
            try:
                result = self.ingest_document(path)
                result["file_path"] = path
                results.append(result)
            except Exception as e:
                results.append({
                    "file_path": path,
                    "status": "error", 
                    "error": str(e)
                })
        return results
    
    async def query(self, question: str):
        """Query the RAG system"""
        # LangChain handles the entire RAG pipeline in one call!
        result = await self.qa_chain.ainvoke({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "unknown") 
                       for doc in result["source_documents"]],
            "retrieved_documents": len(result["source_documents"])
        }

# Even simpler with LCEL (LangChain Expression Language)
def create_lcel_rag_chain():
    """
    Ultra-concise RAG with LangChain Expression Language
    This is ~50 lines vs our ~1000!
    """
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain.prompts import ChatPromptTemplate
    
    # Components
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama3:70b-instruct")
    
    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Answer:
    """)
    
    # Chain with LCEL - this is the entire RAG pipeline!
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain, vectorstore

async def demo_langchain_rag():
    """Demo the LangChain version"""
    print("🔗 LangChain RAG Demo")
    print("=" * 40)
    
    # Method 1: Class-based approach
    pipeline = LangChainRAGPipeline()
    
    # Create sample doc
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        LangChain is a framework for developing applications powered by language models.
        It provides modular components for building RAG systems, agents, and chains.
        LangChain integrates with many vector stores, LLMs, and data sources.
        """)
        doc_path = f.name
    
    # Ingest and query
    result = pipeline.ingest_document(doc_path)
    print(f"📄 Ingested: {result['chunks_created']} chunks")
    
    answer = await pipeline.query("What is LangChain?")
    print(f"🤖 Answer: {answer['answer']}")
    
    # Method 2: LCEL approach (even simpler)
    print(f"\n🔗 LCEL (Ultra-concise) Demo")
    chain, vectorstore = create_lcel_rag_chain()
    
    # Add document to vectorstore
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = splitter.create_documents([open(doc_path).read()], metadatas=[{"source": doc_path}])
    vectorstore.add_documents(docs)
    
    # Query with LCEL chain
    lcel_answer = await chain.ainvoke("What is LangChain used for?")
    print(f"🚀 LCEL Answer: {lcel_answer}")
    
    # Cleanup
    Path(doc_path).unlink()

if __name__ == "__main__":
    print("This would require: pip install langchain langchain-community chromadb")
    print("Run demo with: asyncio.run(demo_langchain_rag())")
