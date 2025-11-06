from abc import ABC, abstractmethod
from typing import Dict, Any
import subprocess

class Tool(ABC):
    """Abstract base class for all tools."""
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def run(self, args: Dict[str, Any]) -> str:
        pass

class GrepTool(Tool):
    """A tool for performing keyword searches using grep."""
    def __init__(self, document_path: str):
        self.document_path = document_path

    def name(self) -> str:
        return "grep"

    def description(self) -> str:
        return "Performs a simple, case-insensitive keyword search over the raw text of the documents. Useful for finding specific terms, names, or acronyms."

    def run(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "Error: The 'query' argument is required for the grep tool."

        try:
            result = subprocess.run(
                ["grep", "-i", query, self.document_path],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except FileNotFoundError:
            return f"Error: The source file '{self.document_path}' was not found."
        except subprocess.CalledProcessError as e:
            return f"Error executing grep: {e.stderr}"
