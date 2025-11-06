from typing import Dict, Any, List
from src.tools.base import Tool
from src.retriever import FAISSRetriever, BM25Retriever
import faiss
import pickle

class FaissSearchTool(Tool):
    """A tool for performing semantic search using FAISS."""
    def __init__(self, index_path: str, embed_model: str, chunks_path: str):
        self.index = faiss.read_index(index_path)
        self.retriever = FAISSRetriever(self.index, embed_model)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def name(self) -> str:
        return "faiss_search"

    def description(self) -> str:
        return "Performs a semantic search using vector embeddings. Ideal for finding conceptually related information, even if the keywords don't match exactly."

    def run(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "Error: The 'query' argument is required for the faiss_search tool."

        scores = self.retriever.get_scores(query, 5, self.chunks)
        top_k_indices = sorted(scores, key=scores.get, reverse=True)[:5]

        return "\n".join([self.chunks[i] for i in top_k_indices])

class BM25ExplorerTool(Tool):
    """A tool for performing keyword relevance search using BM25."""
    def __init__(self, index_path: str, chunks_path: str):
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        self.retriever = BM25Retriever(self.index)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def name(self) -> str:
        return "bm25_explorer"

    def description(self) -> str:
        return "Uses the BM25 algorithm to find documents that are highly relevant to the query's keywords, balancing term frequency and inverse document frequency. Excellent for queries that depend on specific but common terms."

    def run(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "Error: The 'query' argument is required for the bm25_explorer tool."

        scores = self.retriever.get_scores(query, 5, self.chunks)
        top_k_indices = sorted(scores, key=scores.get, reverse=True)[:5]

        return "\n".join([self.chunks[i] for i in top_k_indices])
