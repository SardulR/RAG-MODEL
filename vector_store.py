
from typing import List, Optional, Tuple
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_path: str = "faiss_index"):
        """Initialize vector store with FAISS"""
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.store = None
        self.load_index()

    def get_embedding(self, doc: Document) -> np.ndarray:
        """Get embedding for a document, handling both text and images"""
        try:
            if doc.metadata.get("modality") == "image":
                embedding = doc.metadata.get("embedding")
                if embedding is None:
                    raise ValueError("Image document missing embedding")
                return np.array(embedding)
            else:
                # For text documents, use OpenAI embeddings
                text = doc.page_content or ""
                if not text.strip():
                    logger.warning("Empty text content in document")
                    text = doc.metadata.get("source", "empty document")
                return np.array(self.embeddings.embed_query(text))
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def initialize_store(self, documents: List[Document]) -> None:
        """Initialize the vector store with documents"""
        try:
            if not documents:
                logger.warning("No documents provided for initialization")
                return

            logger.info(f"Initializing store with {len(documents)} documents")
            # Get embeddings for all documents
            embeddings_list = []
            texts = []
            for doc in documents:
                embedding = self.get_embedding(doc)
                embeddings_list.append(embedding)
                texts.append(doc.page_content)

            # Convert to numpy array
            embeddings = np.array(embeddings_list)

            # Initialize FAISS index
            self.store = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)), 
                embedding=self.embeddings
            )
            logger.info("Successfully initialized FAISS vector store")
            self.save_index()
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing store"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return

            logger.info(f"Adding {len(documents)} documents to store")
            if self.store is None:
                self.initialize_store(documents)
            else:
                # Get embeddings for new documents
                embeddings_list = []
                texts = []
                for doc in documents:
                    embedding = self.get_embedding(doc)
                    embeddings_list.append(embedding)
                    texts.append(doc.page_content)

                # Add to FAISS store
                self.store.add_embeddings(
                    text_embeddings=list(zip(texts, embeddings_list)),
                    embedding=self.embeddings
                )
                self.save_index()
            logger.info(f"Successfully added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform similarity search"""
        try:
            if self.store is None:
                logger.warning("Vector store not initialized")
                return []

            logger.info(f"Performing similarity search for query: {query}")
            # Get query embedding
            query_embedding = np.array(self.embeddings.embed_query(query))

            # Search in FAISS store
            results = self.store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} relevant documents")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def save_index(self) -> None:
        """Save the FAISS index"""
        try:
            if self.store:
                self.store.save_local(self.index_path)
                logger.info(f"Saved index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self) -> bool:
        """Load the FAISS index if it exists"""
        try:
            if os.path.exists(self.index_path):
                logger.info(f"Loading index from {self.index_path}")
                self.store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Successfully loaded index from {self.index_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
