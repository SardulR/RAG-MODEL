
from typing import Dict, Any, List, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI

class QueryEngine:
    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)

    def format_context(self, docs: List[Tuple[Document, float]]) -> str:
        """Format documents into context string"""
        if not docs:
            return ""

        formatted_docs = []
        for i, (doc, score) in enumerate(docs):
            # Extract source and content
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            if content:
                formatted_docs.append(f"[{i+1}] From {source}: {content} (Score: {score:.4f})")

        return "\n\n".join(formatted_docs)

    def generate_response(self, query: str, docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Generate response based on query and documents"""
        try:
            context = self.format_context(docs)
            if not context:
                return {
                    "answer": "I don't have enough relevant information to answer that question. Please try asking something else or provide more documents.",
                    "sources": []
                }

            messages = [
                {
                    "role": "system", 
                    "content": """You are a helpful assistant that provides accurate answers based on the given context. 
                    If the context doesn't contain information to answer the question, explicitly state that. 
                    Use the source numbers [n] to cite information. Keep answers clear and concise."""
                },
                {
                    "role": "user",
                    "content": f"Based on the following context, answer this question: {query}\n\nContext:\n{context}"
                }
            ]

            response = self.llm.invoke(messages)

            return {
                "answer": response.content,
                "sources": [
                    {
                        "content": doc[0].page_content,
                        "score": float(doc[1]),
                        "metadata": doc[0].metadata
                    } 
                    for doc in docs
                ]
            }

        except Exception as e:
            return {
                "answer": f"An error occurred while generating the response: {str(e)}",
                "sources": []
            }
