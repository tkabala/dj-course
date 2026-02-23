"""
Tool definitions and execution system for LLM interactions.
Provides a unified interface for defining and executing tools across different LLM clients.
"""

from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from google.genai import types


@dataclass
class ToolDefinition:
    """
    Universal tool definition that can be converted to different LLM formats.

    Attributes:
        name: Tool name (function name)
        description: Description of what the tool does
        parameters: JSON Schema format parameters definition
    """
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format

    def to_gemini_tool(self) -> types.Tool:
        """
        Converts to Gemini tool format.

        Returns:
            types.Tool: Gemini-compatible tool definition
        """
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters=self.parameters
                )
            ]
        )

    def to_openai_function(self) -> Dict[str, Any]:
        """
        Converts to OpenAI function calling format.

        Returns:
            Dict: OpenAI-compatible function definition
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# Define the set_thread_title tool
SET_THREAD_TITLE_TOOL = ToolDefinition(
    name="set_thread_title",
    description="Use this function to set a descriptive title for the current conversation thread when you have enough information to understand the main topic. The title should be 3-7 words in Polish (or the user's language) and capture the essence of what the conversation is about. Call this function when you feel you understand the topic well enough - this could be after the first message or after a few exchanges. Example: if discussing Python programming, use 'Programowanie w Pythonie'.",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "A concise title (3-7 words) that describes the conversation topic. Use the same language as the user."
            }
        },
        "required": ["title"]
    }
)


CLARIFY_USER_QUESTION_TOOL = ToolDefinition(
    name="clarify_user_question",
    description="Ask the user a clarifying question when the request is ambiguous or you need more information to proceed. Use this tool whenever you are in doubt about what the user wants — do not guess. Present a clear question with a list of possible answers for the user to choose from.",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The clarifying question to ask the user."
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of possible answers for the user to choose from (2-6 items)."
            }
        },
        "required": ["question", "options"]
    }
)


class ToolExecutor:
    """
    Executes tool calls and manages tool state.

    Attributes:
        chat_session: ChatSession instance that tools can modify
        mcp_manager: Optional MCPClientManager for routing MCP tool calls
    """

    def __init__(self, chat_session, mcp_manager=None):
        """
        Initialize tool executor with a chat session.

        Args:
            chat_session: ChatSession instance that tools can modify
            mcp_manager: Optional MCPClientManager instance
        """
        self.chat_session = chat_session
        self.mcp_manager = mcp_manager
        self._tool_handlers: Dict[str, Callable] = {
            'set_thread_title': self._handle_set_thread_title,
            'clarify_user_question': self._handle_clarify_user_question,
        }

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a tool by name with given arguments.

        Tool names prefixed with 'mcp__' are routed to the MCP client manager.
        Format: mcp__{server_name}__{tool_name}

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool

        Returns:
            Dictionary with execution result: {"success": bool, "message": str, "data": Any}
        """
        if tool_name.startswith("mcp__"):
            return self._handle_mcp_tool(tool_name, arguments)

        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}",
                "data": None
            }

        try:
            return handler(arguments)
        except Exception as e:
            return {
                "success": False,
                "message": f"Tool execution error: {str(e)}",
                "data": None
            }

    def _handle_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routes an MCP tool call to the MCPClientManager.

        Args:
            tool_name: Namespaced tool name: mcp__{server_name}__{bare_tool_name}
            arguments: Tool arguments

        Returns:
            Execution result dictionary
        """
        if self.mcp_manager is None:
            return {
                "success": False,
                "message": "MCP manager not available",
                "data": None
            }

        # Parse server_name and bare tool_name from the namespaced name
        parts = tool_name.split("__", 2)  # ["mcp", server_name, tool_name]
        if len(parts) != 3:
            return {
                "success": False,
                "message": f"Invalid MCP tool name format: {tool_name}",
                "data": None
            }
        _, server_name, bare_tool_name = parts

        try:
            result = self.mcp_manager.call_tool(server_name, bare_tool_name, arguments)
            return {
                "success": True,
                "message": result.get("content", str(result)),
                "data": result
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"MCP tool execution error: {str(e)}",
                "data": None
            }

    def _handle_set_thread_title(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handler for set_thread_title tool.

        Args:
            arguments: {"title": "Thread Title"}

        Returns:
            Execution result dictionary with success status and message
        """
        title = arguments.get('title', '').strip()

        if not title:
            return {
                "success": False,
                "message": "Title cannot be empty",
                "data": None
            }

        # Set title in chat session
        if self.chat_session.set_title(title):
            # Save to persist the title
            self.chat_session.save_to_file()
            return {
                "success": True,
                "message": f"Ustawiam tytuł sesji: {title}",
                "data": {"title": title}
            }
        else:
            return {
                "success": False,
                "message": "Nie udało się ustawić tytułu sesji",
                "data": None
            }


    def _handle_clarify_user_question(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from cli.console import show_selection_list
        question = arguments.get('question', 'Wybierz opcję:')
        options = arguments.get('options', [])

        if not options:
            return {"success": False, "message": "Brak opcji do wyboru", "data": None}

        selected = show_selection_list(question, options)

        if selected is None:
            return {"success": False, "message": "Użytkownik anulował wybór", "data": None}

        return {
            "success": True,
            "message": f"Użytkownik wybrał: {selected}",
            "data": {"selected": selected}
        }


def should_offer_title_tool(chat_session) -> bool:
    """
    Determines if the title tool should be offered to the LLM.

    The tool should be offered as long as the session doesn't have a title yet.
    The model will decide when it has enough information to set an appropriate title.

    Args:
        chat_session: ChatSession instance

    Returns:
        True if tool should be offered, False otherwise
    """
    # Offer tool as long as there's no title - model decides when to call it
    return not chat_session.has_title()
