# TokenSmith

**TokenSmith** is a local-first database system for students to query textbooks, lecture slides, and notes and get fast, cited answers on their own machines using local LLMs. It is based on an agent-based retrieval augmented generation (RAG) architecture and applies database-inspired principles like query planning and multi-level indexing to optimize the ingestion -> retrieval -> generation pipeline.

## Capabilities

*   **Agent-Based RAG:** An SLM-based agent acts as a "query planner" to choose the best retrieval tool for each query.
*   **Multi-Level Indexing:** Creates indexes for both section summaries and full-text chunks, enabling a two-step, hierarchical retrieval process.
*   **Specialized Retrieval Tools:** Includes tools for semantic search (FAISS), keyword search (BM25), and simple string matching (grep).
*   **Local-First:** All processing is done locally using GGUF models with `llama.cpp`.
*   **Configurable:** The entire pipeline, from chunking to retrieval, is configurable via a single YAML file.

## Requirements

*   **Python** 3.9+
*   **Conda/Miniconda**
*   **System prerequisites**:
    *   macOS: Xcode Command Line Tools
    *   Linux: GCC, make, CMake
    *   Windows: Visual Studio Build Tools

## Quick Start

### 1) Clone the repository and Download the models

```shell
git clone https://github.com/georgia-tech-db/TokenSmith.git
cd TokenSmith
```

Create the model directory and put in the appropriate models in it.
```shell
mkdir models
cd models
```

Now, let's say config.yaml has following configs:
```yaml
embed_model: "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
model_path: "models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
```
For above config file, download appropriate files from the below link
and put them in the `models/` folder with the expected file name.
- https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/tree/main
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main

### 2) Build (creates env, builds llama.cpp, installs deps)

```shell
make build
```

Creates a Conda env `tokensmith`, installs Python deps, and builds/detects `llama.cpp`.

### 3) Activate the environment

```shell
conda activate tokensmith
```

### 4) Prepare documents

```shell
mkdir -p data/chapters
cp your-documents.pdf data/chapters/
```

### 5) Index documents

This step now includes generating summaries for each section, which requires an LLM.

```shell
make run-index
```

### 6) Chat

```shell
python -m src.main chat
```

### 7) Deactivate

```shell
conda deactivate
```

## Configuration

The entire pipeline is configured via `config/config.yaml`.

### Example

```yaml
embed_model: "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
top_k: 5
pool_size: 50
ensemble_method: "borda"
ranker_weights: {"faiss":0.5,"bm25":0.5}
rrf_k: 60
max_gen_tokens: 400
seg_filter: null
chunk_mode : "sections"
model_path: "models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
recursive_chunk_size: 2000
recursive_overlap: 400
```

## Development

```shell
make help
make env
make build-llama
make install
make build
make test
make clean
```
