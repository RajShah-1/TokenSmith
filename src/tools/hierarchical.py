from typing import Dict, Any
from src.tools.base import Tool
from src.retriever import FAISSRetriever, BM25Retriever
from src.generator import answer
import faiss
import pickle

class HierarchicalRetrieverTool(Tool):
    """
    A tool that performs a broad retrieval and then uses an LLM to re-rank and summarize the results.
    """
    def __init__(self, faiss_index_path: str, bm25_index_path: str, embed_model: str, model_path: str, chunks_path: str):
        self.faiss_index = faiss.read_index(faiss_index_path)
        with open(bm25_index_path, "rb") as f:
            self.bm25_index = pickle.load(f)

        self.faiss_retriever = FAISSRetriever(self.faiss_index, embed_model)
        self.bm25_retriever = BM25Retriever(self.bm25_index)
        self.model_path = model_path
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def name(self) -> str:
        return "hierarchical_retriever"

    def description(self) -> str:
        return "A more advanced tool that first retrieves a broad set of documents and then uses a single LLM call to re-rank or summarize them to find the most relevant snippets. This is best for complex, multi-faceted queries that require synthesizing information from multiple sources."

    def run(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "Error: The 'query' argument is required for the hierarchical_retriever tool."

        # 1. Broad Retrieval
        faiss_scores = self.faiss_retriever.get_scores(query, 20, self.chunks)
        bm25_scores = self.bm25_retriever.get_scores(query, 20, self.chunks)

        # Simple combination of scores
        combined_scores = {k: faiss_scores.get(k, 0) + bm25_scores.get(k, 0) for k in set(faiss_scores) | set(bm25_scores)}
        top_k_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:10]

        retrieved_context = "\n".join([self.chunks[i] for i in top_k_indices])

        # 2. LLM Re-ranking and Summarization
        rerank_prompt = f"Based on the following retrieved context, please extract the most relevant information to answer the question: '{query}'.\n\nContext:\n{retrieved_context}"

        # This is the single LLM call within the tool
        summary = answer(
            question=rerank_prompt,
            ranked_chunks=[],  # We provide the context directly in the prompt
            model_path=self.model_path,
            max_tokens=500,
            system_prompt_mode="concise"
        )

        return summary
