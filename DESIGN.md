# DESIGN: Agent-Based RAG with a DBMS-Inspired Planner

This document outlines a refined architecture for the TokenSmith RAG pipeline. The core idea is to treat the retrieval process like a database query planning problem. An SLM-based agent will act as a "query planner," analyzing a user's question and choosing the most efficient "access path" (retrieval tool) to find the answer in our document store.

This design prioritizes structured retrieval, robustness, and predictable latency.

## 1. The Agent as a Query Planner

The agent's reasoning loop is a single-shot, non-iterative process designed for low latency and high predictability. It consists of two main stages:

1.  **Planning (1st LLM Call):** The user's query is provided to the SLM, which has a "menu" of available tools (access paths). The SLM's task is to identify the best tool and formulate the precise arguments needed to execute it. This is analogous to a query optimizer choosing between a full table scan, an index scan, or a more complex join operation. The output is a single, structured JSON object specifying the `tool_name` and its `arguments`.

2.  **Execution & Synthesis (Tool Execution + 2nd LLM Call):** The chosen tool is executed with the specified arguments. The retrieved context is then passed to the SLM in a final call to synthesize a human-readable answer.

This two-call architecture ensures that the system remains within our latency budget while allowing for a sophisticated, query-dependent retrieval strategy.

## 2. The Toolset: Access Paths for Document Retrieval

Our toolset is designed to provide a range of specialized "access paths" into our document collection. All tools will be designed to "fail loudly" by raising exceptions on errors, rather than returning error strings.

### a. `GrepTool`
*   **Description:** A direct, substring-matching tool. This is our "full table scan" and is most effective for finding specific, literal strings, such as error messages, function names, or unique identifiers.
*   **Arguments:** `query` (string)
*   **Returns:** A formatted string of matching lines.

### b. `FaissSearchTool`
*   **Description:** A semantic vector search tool. This is our "primary index scan" for conceptual queries. It's best for understanding user intent and finding information that is thematically related, even if the keywords don't match.
*   **Arguments:** `query` (string)
*   **Returns:** A formatted string of the top-k most semantically similar text chunks.

### c. `BM25ExplorerTool`
*   **Description:** A keyword-based relevance search tool. This is our "secondary index scan," ideal for queries that rely on specific but potentially common terms where TF-IDF is a strong signal.
*   **Arguments:** `query` (string)
*   **Returns:** A formatted string of the top-k most relevant text chunks.

### d. `SectionRetrieverTool` (New Primary Advanced Tool)
*   **Description:** A two-step, hierarchical retrieval tool inspired by multi-level database indexes. It's designed for complex queries that are best answered by first identifying a relevant document section and then pinpointing the exact information within it.
*   **Process:**
    1.  **Step 1 (Summary Scan):** Performs a semantic search over an index of pre-computed *section summaries*. This quickly narrows down the search space to the most relevant section of the document.
    2.  **Step 2 (Sentence-Window Scan):** Within the identified section, it performs a second, localized search to find the most relevant sentence window.
*   **Arguments:** `query` (string)
*   **Returns:** A formatted string containing both the summary of the section and the precise sentence window, providing the LLM with both broad context and specific details.

## 3. The Indexing Pipeline: Pre-computing for Performance

To support our new `SectionRetrieverTool`, the indexing process will be refactored to pre-compute the necessary data structures. This is analogous to a database's `CREATE INDEX` command.

The new indexing pipeline will:
1.  **Extract Sections:** As before, the raw text will be divided into its constituent sections (e.g., chapters or subheadings).
2.  **Generate Section Summaries (New Step):** For each section, a single LLM call will be made to generate a concise summary.
3.  **Create Two Indexes:**
    *   **Summaries Index:** A FAISS index will be created for the vector embeddings of the section summaries.
    *   **Full-Text Index:** A separate FAISS index and a BM25 index will be created for the full-text chunks of the document, as before.

This upfront computational cost during indexing will enable our agent to perform highly efficient, two-step retrievals at query time.
