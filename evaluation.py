
import logging
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def generate_evaluation_report(query: str, response: Dict[str, Any], documents: List[str]) -> Dict[str, Any]:
    """Generate evaluation metrics for RAG response"""
    try:
        evaluator = ChatOpenAI(model="gpt-4", temperature=0)
        
        eval_prompt = f"""
        Evaluate the following RAG response based on these criteria:
        Query: {query}
        Response: {response['answer']}
        Source Documents: {documents}
        
        Rate each metric from 0.0 to 1.0:
        1. Relevance: How well does the response address the query?
        2. Completeness: Does the response include all necessary information?
        3. Coherence: Is the response well-structured and logical?
        
        Format: Return only a JSON object with metrics and final_score
        """
        
        eval_response = evaluator.invoke(eval_prompt)
        metrics = {
            "relevance": float(eval_response.content.get("relevance", 0.0)),
            "completeness": float(eval_response.content.get("completeness", 0.0)),
            "coherence": float(eval_response.content.get("coherence", 0.0))
        }
        
        final_score = sum(metrics.values()) / len(metrics)
        
        return {
            "metrics": metrics,
            "final_score": final_score
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {
            "metrics": {"relevance": 0.0, "completeness": 0.0, "coherence": 0.0},
            "final_score": 0.0
        }
