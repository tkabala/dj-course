import uuid
from typing import List, Any, Union
import os
from files import session_files
from files.wal import append_to_wal
from llm.gemini_client import GeminiLLMClient
from llm.llama_client import LlamaClient
from llm.openai_client import OpenAIClient
from assistant import Assistant
from cli import console

# Context token limit

# Engine to Client Class mapping
ENGINE_MAPPING = {
    'LLAMA_CPP': LlamaClient,
    'GEMINI': GeminiLLMClient,
    'OPENAI': OpenAIClient,
}


class ChatSession:
    """
    Manages everything related to a single chat session.
    Encapsulates session ID, conversation history, assistant, and LLM chat session.
    """
    
    def __init__(self, assistant: Assistant, session_id: str | None = None, history: List[Any] | None = None, title: str | None = None):
        """
        Initialize a chat session.

        Args:
            assistant: Assistant instance that defines the behavior and model for this session
            session_id: Unique session identifier. If None, generates a new UUID.
            history: Initial conversation history. If None, starts empty.
            title: Optional title for the session thread.
        """
        self.assistant = assistant
        self.session_id = session_id or str(uuid.uuid4())
        self._history = history or []
        self._title = title
        self._llm_client: Union[GeminiLLMClient, LlamaClient, OpenAIClient, None] = None
        self._llm_chat_session = None
        self._max_context_tokens = 32768
        self._mcp_manager = None
        self._mcp_tools: List[Any] = []
        self._initialize_llm_session()
        self._initialize_mcp()
    
    def _initialize_llm_session(self):
        """
        Creates or recreates the LLM chat session with current history.
        This should be called after any history modification.
        """
        # Walidacja zmiennej ENGINE
        engine = os.getenv('ENGINE', 'GEMINI').upper()
        if engine not in ENGINE_MAPPING:
            valid_engines = ', '.join(ENGINE_MAPPING.keys())
            raise ValueError(f"ENGINE musi być jedną z wartości: {valid_engines}, otrzymano: {engine}")
        
        # Initialize LLM client if not already created
        if self._llm_client is None:
            SelectedClientClass = ENGINE_MAPPING.get(engine, GeminiLLMClient)
            console.print_info(SelectedClientClass.preparing_for_use_message())
            self._llm_client = SelectedClientClass.from_environment()
            console.print_info(self._llm_client.ready_for_use_message())
        
        self._llm_chat_session = self._llm_client.create_chat_session(
            system_instruction=self.assistant.system_prompt,
            history=self._history,
            thinking_budget=0
        )
    
    
    def _initialize_mcp(self):
        """
        Initialise the MCP client manager if mcp_servers.json exists and is non-empty.
        Fetches tool list once and caches it for the session lifetime.
        """
        from azor_mcp.client import MCPClientManager, MCP_CONFIG_PATH
        import os
        if not os.path.exists(MCP_CONFIG_PATH):
            return
        try:
            manager = MCPClientManager()
            if not manager.has_servers():
                return
            self._mcp_manager = manager
            self._mcp_tools = manager.get_all_tools()
            console.print_info(f"🔌 MCP: załadowano {len(self._mcp_tools)} tool(s) z {len(manager._server_configs)} serwera(ów)")
        except Exception as e:
            console.print_info(f"⚠️  MCP: nie udało się zainicjować klienta: {e}")

    @classmethod
    def load_from_file(cls, assistant: Assistant, session_id: str) -> tuple['ChatSession | None', str | None]:
        """
        Loads a session from disk.

        Args:
            assistant: Assistant instance to use for this session
            session_id: ID of the session to load

        Returns:
            tuple: (ChatSession object or None, error_message or None)
        """
        history, title, error = session_files.load_session_history(session_id)

        if error:
            return None, error

        session = cls(assistant=assistant, session_id=session_id, history=history, title=title)
        return session, None
    
    def save_to_file(self) -> tuple[bool, str | None]:
        """
        Saves this session to disk.
        Only saves if history has at least one complete exchange.

        Returns:
            tuple: (success: bool, error_message: str | None)
        """
        # Sync history from LLM session before saving
        if self._llm_chat_session:
            self._history = self._llm_chat_session.get_history()

        return session_files.save_session_history(
            self.session_id,
            self._history,
            self.assistant.system_prompt,
            self._llm_client.get_model_name(),
            self._title
        )
    
    def send_message(self, text: str):
        """
        Sends a message to the LLM and returns the response.
        Updates internal history automatically and logs to WAL.
        Handles tool calls if the LLM requests them.

        Args:
            text: User's message

        Returns:
            Response object from LLM
        """
        if not self._llm_chat_session:
            raise RuntimeError("LLM session not initialized")

        # Import tools
        from llm.tools import should_offer_title_tool, ToolExecutor, SET_THREAD_TITLE_TOOL, CLARIFY_USER_QUESTION_TOOL

        # Build tool list: title tool (when needed) + clarify tool + MCP tools
        active_tools = []
        if should_offer_title_tool(self):
            active_tools.append(SET_THREAD_TITLE_TOOL)
        active_tools.append(CLARIFY_USER_QUESTION_TOOL)  # Always offer
        active_tools.extend(self._mcp_tools)

        # Recreate session with tools if any are active
        if active_tools:
            tool_parts = []
            if should_offer_title_tool(self):
                tool_parts.append("🔧 tytuł wątku")
            tool_parts.append("❓ doprecyzowanie")
            if self._mcp_tools:
                tool_parts.append(f"🔌 {len(self._mcp_tools)} MCP tool(s)")
            console.print_info(f"Oferuję tools: {' | '.join(tool_parts)}")
            self._llm_chat_session = self._llm_client.create_chat_session(
                system_instruction=self.assistant.system_prompt,
                history=self._history,
                thinking_budget=0,
                tools=active_tools
            )

        # Send message
        response = self._llm_chat_session.send_message(text)

        # Handle tool calls (loop to support multi-turn: model may call tools multiple times)
        tool_executor = ToolExecutor(self, mcp_manager=self._mcp_manager)
        while hasattr(response, 'has_tool_calls') and response.has_tool_calls():
            console.print_info(f"📞 Model wywołał {len(response.tool_calls)} tool(s):")

            RESULT_RETURNING_TOOLS = {'clarify_user_question'}
            has_result_returning_calls = any(
                tc['name'].startswith('mcp__') or tc['name'] in RESULT_RETURNING_TOOLS
                for tc in response.tool_calls
            )
            results_for_model = []

            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['arguments']
                tool_id = tool_call.get('id')

                console.print_info(f"  - Tool: {tool_name} | Argumenty: {tool_args}")

                result = tool_executor.execute_tool(tool_name, tool_args)

                if result['success']:
                    msg_lines = str(result['message']).splitlines()
                    truncated = "\n    ".join(msg_lines[:3])
                    suffix = " ..." if len(msg_lines) > 3 else ""
                    console.print_info(f"  ✓ {truncated}{suffix}")
                else:
                    console.print_error(f"  ✗ {result['message']}")

                if tool_name.startswith('mcp__') or tool_name in RESULT_RETURNING_TOOLS:
                    results_for_model.append({
                        "id": tool_id,
                        "name": tool_name,
                        "result": result
                    })

            if has_result_returning_calls:
                # Proper tool-result protocol: send results back, model may call more tools
                console.print_info(f"🔄 Odsyłam wyniki do modelu...")
                response = self._llm_chat_session.send_tool_results(results_for_model)
            else:
                # Fire-and-forget (e.g. set_thread_title): resend original message without tools
                console.print_info(f"🔄 Ponownie wysyłam wiadomość bez tools aby uzyskać odpowiedź...")
                self._history = self._llm_chat_session.get_history()
                self._llm_chat_session = self._llm_client.create_chat_session(
                    system_instruction=self.assistant.system_prompt,
                    history=self._history,
                    thinking_budget=0,
                    tools=None
                )
                response = self._llm_chat_session.send_message(text)
                break  # After fire-and-forget, the response will not have tool calls

        # Sync history
        self._history = self._llm_chat_session.get_history()

        # Log to WAL
        total_tokens = self.count_tokens()
        success, error = append_to_wal(
            session_id=self.session_id,
            prompt=text,
            response_text=response.text,
            total_tokens=total_tokens,
            model_name=self._llm_client.get_model_name()
        )

        if not success and error:
            # We don't want to fail the entire message sending because of WAL issues
            # Just log the error to stderr or similar - but for now we'll silently continue
            pass

        return response
    
    def get_history(self) -> List[Any]:
        """Returns the current conversation history."""
        # Always sync from LLM session to ensure consistency
        if self._llm_chat_session:
            self._history = self._llm_chat_session.get_history()
        return self._history
    
    def clear_history(self):
        """Clears all conversation history and reinitializes the LLM session."""
        self._history = []
        self._initialize_llm_session()
        self.save_to_file()
    
    def pop_last_exchange(self) -> bool:
        """
        Removes the last user-assistant exchange from history.
        
        Returns:
            bool: True if successful, False if insufficient history
        """
        current_history = self.get_history()
        
        if len(current_history) < 2:
            return False
        
        # Remove last 2 entries (user + assistant)
        self._history = current_history[:-2]
        
        # Reinitialize LLM session with modified history
        self._initialize_llm_session()
        
        self.save_to_file()
        
        return True
    
    def count_tokens(self) -> int:
        """
        Counts total tokens in the conversation history.
        
        Returns:
            int: Total token count
        """
        if not self._llm_client:
            return 0
        return self._llm_client.count_history_tokens(self._history)
    
    def is_empty(self) -> bool:
        """
        Checks if session has any complete exchanges.
        
        Returns:
            bool: True if history has less than 2 entries
        """
        return len(self._history) < 2
    
    def get_remaining_tokens(self) -> int:
        """
        Calculates remaining tokens based on context limit.
        
        Returns:
            int: Remaining token count
        """
        total = self.count_tokens()
        return self._max_context_tokens - total
    
    def get_token_info(self) -> tuple[int, int, int]:
        """
        Gets comprehensive token information for this session.
        
        Returns:
            tuple: (total_tokens, remaining_tokens, max_tokens)
        """
        total_tokens = self.count_tokens()
        remaining_tokens = self._max_context_tokens - total_tokens
        max_tokens = self._max_context_tokens
        return total_tokens, remaining_tokens, max_tokens
    
    @property
    def assistant_name(self) -> str:
        """
        Gets the display name of the assistant.

        Returns:
            str: The assistant's display name
        """
        return self.assistant.name

    @property
    def title(self) -> str | None:
        """
        Gets the current thread title.

        Returns:
            str | None: The session title or None if not set
        """
        return self._title

    def set_title(self, title: str) -> bool:
        """
        Sets the thread title.

        Args:
            title: The title to set for this session

        Returns:
            bool: True if successful, False if title is empty
        """
        if not title or not title.strip():
            return False
        self._title = title.strip()
        return True

    def has_title(self) -> bool:
        """
        Checks if the session has a title.

        Returns:
            bool: True if title is set, False otherwise
        """
        return self._title is not None and len(self._title) > 0