import json
from typing import List, Dict, Any
from src.tools.base import Tool
from src.generator import answer

class AgenticOrchestrator:
    """
    The core of the agent-based RAG system. It orchestrates the entire process,
    from prompting the SLM for a tool-use plan to executing the chosen tool.
    """
    def __init__(self, tools: List[Tool], model_path: str):
        self.tools = {tool.name(): tool for tool in tools}
        self.model_path = model_path

    def _format_tool_prompt(self) -> str:
        """Formats the list of available tools for the planning prompt."""
        prompt = "You have access to the following tools:\n\n"
        for tool in self.tools.values():
            prompt += f"- **{tool.name()}**: {tool.description()}\n"
        return prompt

    def _get_planning_prompt(self, query: str) -> str:
        """Creates the full planning prompt for the SLM."""
        tool_prompt = self._format_tool_prompt()
        return (
            f"{tool_prompt}\n"
            "Based on the user's query, you must select the single best tool to answer it. "
            "You must respond with a JSON object containing the chosen 'tool_name' and its 'arguments'.\n\n"
            f"User Query: \"{query}\""
        )

    def run(self, query: str) -> str:
        """
        Executes the full agentic loop: plan, execute, and return the result.
        """
        # 1. Planning (1st LLM Call)
        planning_prompt = self._get_planning_prompt(query)
        plan_json_str = answer(
            question=planning_prompt,
            ranked_chunks=[],
            model_path=self.model_path,
            max_tokens=200,
            system_prompt_mode="concise"
        )

        try:
            plan = json.loads(plan_json_str)
            tool_name = plan.get("tool_name")
            tool_args = plan.get("arguments", {})
        except (json.JSONDecodeError, AttributeError):
            return f"Error: The LLM's plan was not valid JSON. Response: {plan_json_str}"

        # 2. Tool Execution
        if tool_name not in self.tools:
            return f"Error: The LLM chose a tool that does not exist: {tool_name}"

        tool = self.tools[tool_name]
        try:
            result = tool.run(tool_args)
            return result
        except Exception as e:
            return f"Error executing the '{tool_name}' tool: {e}"
