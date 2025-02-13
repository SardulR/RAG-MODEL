
import os
import logging
import time
from typing import List, Dict, Any, Optional

from document_processor import DocumentProcessor
from image_handler import ImageHandler
from media_processor import MediaProcessor
from vector_store import VectorStore
from query_engine import QueryEngine
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalRAG:
    def __init__(self, index_path: str = "faiss_index"):
        """Initialize the MultimodalRAG system with its components"""
        logger.info("Initializing MultimodalRAG system...")
        try:
            self.document_processor = DocumentProcessor()
            self.image_handler = ImageHandler()
            self.media_processor = MediaProcessor()
            self.vector_store = VectorStore(index_path)
            self.query_engine = QueryEngine()

            # Define the RAG chain prompt template
            self.prompt_template = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use two sentences maximum and keep the answer concise.

            Question: {question} 
            Context: {context} 
            Answer:
            """)

            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Error initializing MultimodalRAG: {e}")
            raise

    def process_documents(self, file_paths: List[str]) -> None:
        """Process multiple documents and store in vector store"""
        if not file_paths:
            logger.warning("No file paths provided for processing")
            return

        start_time = time.time()
        all_docs = []

        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"File {path} not found")
                continue

            logger.info(f"Processing {path}...")
            try:
                ext = os.path.splitext(path)[1].lower()

                if ext in [".txt", ".md", ".pdf"]:
                    docs = self.document_processor.process_text_file(path)
                    logger.info(f"Processed text file {path}, got {len(docs)} documents")
                elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                    doc = self.image_handler.process_image(path)
                    docs = [doc] if doc else []
                    logger.info(f"Processed image file {path}, got {len(docs)} documents")
                elif ext in [".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".m4a", ".ogg"]:
                    doc = self.media_processor.process_media_file(path)
                    docs = [doc] if doc else []
                    logger.info(f"Processed media file {path}, got {len(docs)} documents")
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    continue

                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Error processing file {path}: {e}")
                continue

        if all_docs:
            logger.info(f"Processing {len(all_docs)} total documents")
            try:
                # Split documents into chunks
                split_docs = self.document_processor.split_documents(all_docs)
                logger.info(f"Split into {len(split_docs)} chunks")

                # Add to vector store
                self.vector_store.add_documents(split_docs)
                logger.info("Successfully added documents to vector store")

                process_time = time.time() - start_time
                logger.info(f"Total processing time: {process_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in document processing pipeline: {e}")
                raise
        else:
            logger.warning("No valid documents were processed")

    def query(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """Query the RAG system with improved chain processing"""
        logger.info(f"Processing query: {query_text}")
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query_text, k=k)
            logger.info(f"Retrieved {len(docs)} relevant documents")

            # Aggregate context from retrieved documents
            context = "\n\n".join([doc[0].page_content for doc in docs])

            # Build the RAG chain
            rag_chain = (
                {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
                | self.prompt_template
                | self.query_engine.llm
                | StrOutputParser()
            )

            # Generate response
            answer = rag_chain.invoke({
                "question": query_text,
                "context": context
            })

            response = {
                "answer": answer,
                "sources": [
                    {
                        "content": doc[0].page_content,
                        "score": doc[1],
                        "metadata": doc[0].metadata
                    } 
                    for doc in docs
                ]
            }

            logger.info("Successfully generated response")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "An error occurred while processing your query.",
                "sources": [],
                "error": str(e)
            }
