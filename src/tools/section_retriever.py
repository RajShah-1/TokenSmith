from typing import Dict, Any
import re
import faiss
import pickle
import numpy as np
from src.tools.base import Tool
from src.embedder import SentenceTransformer

class SectionRetrieverTool(Tool):
    """
    A tool that performs a two-step retrieval process: first on section summaries,
    then on sentence windows within the best section.
    """
    def __init__(self, summaries_index_path: str, chunks_index_path: str, summaries_path: str, chunks_path: str, section_to_chunks_path: str, embed_model: str):
        self.summaries_index = faiss.read_index(summaries_index_path)
        self.chunks_index = faiss.read_index(chunks_index_path)
        with open(summaries_path, "rb") as f:
            self.summaries = pickle.load(f)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(section_to_chunks_path, "rb") as f:
            self.section_to_chunks = pickle.load(f)
        self.embedder = SentenceTransformer(embed_model)

    def name(self) -> str:
        return "section_retriever"

    def description(self) -> str:
        return "A two-step, hierarchical retrieval tool. Best for complex queries that are best answered by first identifying a relevant document section and then pinpointing the exact information within it."

    def run(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            raise ValueError("The 'query' argument is required for the section_retriever tool.")

        # Step 1: Find the most relevant section summary
        q_vec = self.embedder.encode([query]).astype("float32")
        _, top_k_indices = self.summaries_index.search(q_vec, 1)
        best_section_index = top_k_indices[0][0]
        best_section_summary = self.summaries[best_section_index]

        # Step 2: Find the most relevant sentence window in the corresponding section's full text
        chunk_indices = self.section_to_chunks.get(best_section_index, [])
        if not chunk_indices:
            return f"Section Summary:\n{best_section_summary}\n\nNo chunks found for the best section."

        section_text = " ".join([self.chunks[i] for i in chunk_indices])
        sentences = re.split(r'(?<=[.!?]) +', section_text)
        if not sentences or len(sentences) < 3:
            return f"Section Summary:\n{best_section_summary}\n\nRelevant Sentence Window:\n{best_section_text}"

        # Adaptively determine the window size
        num_sentences = len(sentences)
        window_size = max(3, num_sentences // 10)

        # Create sliding windows of sentences
        windows = [" ".join(sentences[i:i+window_size]) for i in range(num_sentences - window_size + 1)]
        if not windows:
            return f"Section Summary:\n{best_section_summary}\n\nRelevant Sentence Window:\n{best_section_text}"

        # Embed all windows
        window_embeddings = self.embedder.encode(windows, batch_size=32)

        # Find the most similar window to the query
        similarities = window_embeddings @ q_vec.T
        best_window_index = np.argmax(similarities)
        best_window = windows[best_window_index]

        return f"Section Summary:\n{best_section_summary}\n\nRelevant Sentence Window:\n{best_window}"
