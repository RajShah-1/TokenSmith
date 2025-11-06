import argparse
import pathlib
import sys
from typing import Dict, Optional

from src.config import QueryPlanConfig
from src.generator import answer
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger, RunLogger

from src.tools.base import GrepTool
from src.tools.retrieval import FaissSearchTool, BM25ExplorerTool
from src.tools.hierarchical import HierarchicalRetrieverTool
from src.planning.orchestrator import AgenticOrchestrator
from src.preprocessing.chunking import DocumentChunker
from src.retriever import load_artifacts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Welcome to TokenSmith!"
    )

    # Required arguments
    parser.add_argument(
        "mode",
        choices=["index", "chat"],
        help="operation mode: 'index' to build index, 'chat' to query"
    )

    # Common arguments
    parser.add_argument(
        "--pdf_dir",
        default="data/chapters/",
        help="directory containing PDF files (default: %(default)s)"
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="prefix for generated index files (default: %(default)s)"
    )
    parser.add_argument(
        "--model_path",
        help="path to generation model (uses config default if not specified)"
    )
    parser.add_argument(
        "--system_prompt_mode",
        choices=["baseline", "tutor", "concise", "detailed"],
        default="baseline",
        help="system prompt mode (choices: baseline, tutor, concise, detailed)"
    )
    
    # Indexing-specific arguments
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument(
        "--pdf_range",
        metavar="START-END",
        help="specific range of PDFs to index (e.g., '27-33')"
    )
    indexing_group.add_argument(
        "--keep_tables",
        action="store_true",
        help="include tables in the index"
    )
    indexing_group.add_argument(
        "--visualize",
        action="store_true",
        help="generate visualizations during indexing"
    )

    return parser.parse_args()


def run_index_mode(args: argparse.Namespace, cfg: QueryPlanConfig):
    """Handles the logic for building the index."""

    # Robust range filtering
    try:
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end + 1)] # Inclusive range
            print(f"Indexing PDFs in range: {start}-{end}")
        else:
            pdf_paths = None
    except ValueError:
        print(f"ERROR: Invalid format for --pdf_range. Expected 'start-end', but got '{args.pdf_range}'.")
        sys.exit(1)
    
    strategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    
    artifacts_dir = cfg.make_artifacts_directory()

    build_index(
        markdown_file="data/book_without_image.md",
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
    )


def get_final_answer(query: str, context: str, model_path: str, max_tokens: int, system_prompt_mode: str) -> str:
    """
    Synthesizes the final answer using the retrieved context.
    """
    synthesis_prompt = (
        f"Based on the following context, please provide a comprehensive answer to the user's query.\n\n"
        f"Context:\n{context}\n\n"
        f"Query: {query}"
    )
    
    return answer(
        question=synthesis_prompt,
        ranked_chunks=[],
        model_path=model_path,
        max_tokens=max_tokens,
        system_prompt_mode=system_prompt_mode
    )

def run_chat_session(args: argparse.Namespace, cfg: QueryPlanConfig):
    """
    Initializes artifacts and runs the main interactive chat loop.
    """
    logger = get_logger()

    print("Welcome to Tokensmith! Initializing agent...")
    try:
        artifacts_dir = cfg.make_artifacts_directory()

        # Initialize tools
        chunks_path = str(artifacts_dir / f"{args.index_prefix}_chunks.pkl")
        tools = [
            GrepTool(document_path="data/book_without_image.md"),
            FaissSearchTool(
                index_path=str(artifacts_dir / f"{args.index_prefix}.faiss"),
                embed_model=cfg.embed_model,
                chunks_path=chunks_path
            ),
            BM25ExplorerTool(
                index_path=str(artifacts_dir / f"{args.index_prefix}_bm25.pkl"),
                chunks_path=chunks_path
            ),
            HierarchicalRetrieverTool(
                faiss_index_path=str(artifacts_dir / f"{args.index_prefix}.faiss"),
                bm25_index_path=str(artifacts_dir / f"{args.index_prefix}_bm25.pkl"),
                embed_model=cfg.embed_model,
                model_path=args.model_path or cfg.model_path,
                chunks_path=chunks_path
            )
        ]
        
        orchestrator = AgenticOrchestrator(tools=tools, model_path=args.model_path or cfg.model_path)

    except Exception as e:
        print(f"ERROR: Failed to initialize chat artifacts: {e}")
        print("Please ensure you have run 'index' mode first.")
        sys.exit(1)

    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            logger.log_query_start(q)
            start_time = time.time()

            # 1. Use the orchestrator to get the retrieved context
            retrieved_context = orchestrator.run(q)

            # 2. Synthesize the final answer
            ans = get_final_answer(
                query=q,
                context=retrieved_context,
                model_path=args.model_path or cfg.model_path,
                max_tokens=cfg.max_gen_tokens,
                system_prompt_mode=args.system_prompt_mode or cfg.system_prompt_mode
            )

            end_time = time.time()

            print("\n=================== START OF ANSWER ===================")
            print(ans.strip() if ans and ans.strip() else "(No output from model)")
            print("\n==================== END OF ANSWER ====================")
            logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": args.model_path or cfg.model_path})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logger.log_error(str(e))
            break


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Config loading
    config_path = pathlib.Path("config/config.yaml")
    cfg = None
    if config_path.exists():
        cfg = QueryPlanConfig.from_yaml(config_path)

    if cfg is None:
        raise FileNotFoundError(
            "No config file provided and no fallback found at config/ or ~/.config/tokensmith/"
        )

    init_logger(cfg)

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()
