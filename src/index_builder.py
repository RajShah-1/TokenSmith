#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, cfg, keep_tables=True, do_visualize=False)
"""

import os
import pickle
import pathlib
import re
from typing import List, Dict

import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer
from src.generator import answer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown
from src.config import QueryPlanConfig


# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

# Default keywords to exclude sections
DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']

# ------------------------ Main index builder -----------------------------

def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str, 
    do_visualize: bool = False,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []
    section_to_chunks: Dict[int, List[int]] = {}

    # Extract sections from markdown. Exclude some with certain
    # keywords if required.
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )

    # Step 1: Chunk using DocumentChunker
    for i, c in enumerate(sections):
        has_table = bool(TABLE_RE.search(c['content']))
        meta = {
            "filename": markdown_file,
            "chunk_id": i,
            "mode": chunk_config.to_string(),
            "keep_tables": chunker.keep_tables,
            "char_len": len(c['content']),
            "word_len": len(c['content'].split()),
            "has_table": has_table,
            "section": c['heading'], 
            "text_preview": c['content'][:100]
        }
        
        # Use DocumentChunker to recursively split this section
        sub_chunks = chunker.chunk(c['content'])
        chunk_indices = []
        for sub_c in sub_chunks:
            chunk_indices.append(len(all_chunks))
            all_chunks.append(sub_c)
            sources.append(markdown_file)
            metadata.append(meta)
        section_to_chunks[i] = chunk_indices

    # Step 2: Generate Section Summaries
    print(f"Generating summaries for {len(sections)} sections...")
    section_summaries = []
    for section in sections:
        summary_prompt = f"Summarize the following section in one or two sentences:\n\n{section['content']}"
        summary = answer(
            question=summary_prompt,
            ranked_chunks=[],
            model_path=model_path,
            max_tokens=100,
            system_prompt_mode="concise"
        )
        section_summaries.append(summary)

    # Step 3: Create embeddings for summaries and chunks
    print(f"Embedding {len(all_chunks):,} chunks and {len(section_summaries)} summaries with {pathlib.Path(embedding_model_path).stem} ...")
    embedder = SentenceTransformer(embedding_model_path)

    chunk_embeddings = embedder.encode(
        all_chunks, batch_size=4, show_progress_bar=True
    )
    summary_embeddings = embedder.encode(
        section_summaries, batch_size=4, show_progress_bar=True
    )

    # Step 4: Build FAISS indexes
    print(f"Building FAISS indexes...")
    chunk_dim = chunk_embeddings.shape[1]
    summary_dim = summary_embeddings.shape[1]

    chunk_index = faiss.IndexFlatL2(chunk_dim)
    chunk_index.add(chunk_embeddings)
    faiss.write_index(chunk_index, str(artifacts_dir / f"{index_prefix}_chunks.faiss"))

    summary_index = faiss.IndexFlatL2(summary_dim)
    summary_index.add(summary_embeddings)
    faiss.write_index(summary_index, str(artifacts_dir / f"{index_prefix}_summaries.faiss"))
    print(f"FAISS indexes built successfully.")

    # Step 5: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")

    # Step 6: Dump index artifacts
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_summaries.pkl", "wb") as f:
        pickle.dump(section_summaries, f)
    with open(artifacts_dir / f"{index_prefix}_section_to_chunks.pkl", "wb") as f:
        pickle.dump(section_to_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")

    # # Step 6: Optional visualization
    if do_visualize:
        visualize(embeddings, sources)


# ------------------------ Helper functions ------------------------------

def preprocess_for_bm25(text: str) -> list[str]:
    """
    Simplifies text to keep only letters, numbers, underscores, hyphens,
    apostrophes, plus, and hash — suitable for BM25 tokenization.
    """
    # Convert to lowercase
    text = text.lower()

    # Keep only allowed characters
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)

    # Split by whitespace
    tokens = text.split()

    return tokens


def visualize(embeddings, sources):
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        red = PCA(n_components=2).fit_transform(embeddings)
        uniq = sorted(set(sources))
        cmap = {s: i for i, s in enumerate(uniq)}
        colors = [cmap[s] for s in sources]

        plt.figure(figsize=(10, 7))
        sc = plt.scatter(red[:, 0], red[:, 1], c=colors, cmap="tab10", alpha=0.55)
        plt.title("Vector index (PCA)")
        plt.legend(
            handles=sc.legend_elements()[0],
            labels=uniq,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[visualize] skipped ({e})")
