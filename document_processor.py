
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        return " ".join(text.split())

    def process_text_file(self, file_path: str) -> List[Document]:
        """Process text files and return documents"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            text = self.clean_text(text)
            if text:
                return [Document(
                    page_content=text,
                    metadata={"source": file_path, "category": "text", "modality": "text"}
                )]
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
        return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
