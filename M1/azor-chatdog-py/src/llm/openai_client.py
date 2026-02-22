"""
OpenAI-Compatible LLM Client Implementation
Encapsulates all OpenAI API interactions, compatible with Ollama, llama-server, and other OpenAI-compatible services.
"""

import os
from typing import Optional, List, Any, Dict
from openai import OpenAI
from dotenv import load_dotenv
from cli import console
from .openai_validation import OpenAIConfig

class OpenAIChatSession:
    """
    Wrapper class that provides a chat session interface compatible with Gemini's interface.
    Manages conversation history and provides send_message() and get_history() methods.
    """

    def __init__(self, openai_client: OpenAI, model_name: str, system_instruction: str, history: Optional[List[Dict]] = None,
                 temperature: float = 0.7, top_p: float = 1.0, tools: Optional[List] = None):
        """
        Initialize the OpenAI chat session.

        Args:
            openai_client: Initialized OpenAI client instance
            model_name: Model name to use for completions
            system_instruction: System prompt for the assistant
            history: Previous conversation history in universal format
            temperature: Controls randomness (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            tools: Optional list of ToolDefinition objects for function calling
        """
        self.openai_client = openai_client
        self.model_name = model_name
        self.system_instruction = system_instruction
        self._history = history or []
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools
        self._pending_tool_calls = None  # Raw OpenAI tool_calls for proper result flow

    def send_message(self, text: str) -> Any:
        """
        Sends a message to the OpenAI model and returns a response object.

        Args:
            text: User's message

        Returns:
            Response object with .text attribute or tool calls
        """
        # Add user message to history (universal format)
        user_message = {"role": "user", "parts": [{"text": text}]}
        self._history.append(user_message)

        # Convert universal format to OpenAI format for API call
        openai_messages = self._build_openai_messages()

        try:
            # Build API parameters
            api_params = {
                "model": self.model_name,
                "messages": openai_messages,
                "temperature": self.temperature,
                "top_p": self.top_p
            }

            # Add tools if provided
            if self.tools:
                api_params["tools"] = [tool.to_openai_function() for tool in self.tools]

            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(**api_params)

            # Check for tool calls
            if response.choices[0].message.tool_calls:
                # Save raw tool_calls for the tool-result follow-up
                self._pending_tool_calls = response.choices[0].message.tool_calls

                # Extract tool calls and return special response
                import json
                tool_calls = []
                for tc in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    })

                # Return response with tool calls
                return OpenAIResponseWithToolCalls(tool_calls)
            else:
                # Normal text response
                response_text = response.choices[0].message.content.strip()

                # Add assistant response to history (universal format)
                assistant_message = {"role": "model", "parts": [{"text": response_text}]}
                self._history.append(assistant_message)

                # Return response object compatible with Gemini interface
                return OpenAIResponse(response_text)

        except Exception as e:
            console.print_error(f"Błąd podczas generowania odpowiedzi OpenAI: {e}")
            # Return error response
            error_text = "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."
            assistant_message = {"role": "model", "parts": [{"text": error_text}]}
            self._history.append(assistant_message)
            return OpenAIResponse(error_text)

    def get_history(self) -> List[Dict]:
        """Returns the current conversation history in universal format."""
        return self._history

    def send_tool_results(self, tool_results: list) -> Any:
        """
        Feed tool results back to the model using the proper OpenAI tool protocol.

        Args:
            tool_results: list of {"id": str, "name": str, "result": dict}

        Returns:
            Response object with .text attribute
        """
        messages = self._build_openai_messages()

        # Re-add the assistant's tool_call message that wasn't stored in _history
        if self._pending_tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in self._pending_tool_calls
                ]
            })

        # Add tool result messages
        for tr in tool_results:
            content = tr["result"].get("message", str(tr["result"]))
            messages.append({
                "role": "tool",
                "tool_call_id": tr["id"],
                "content": content
            })

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            console.print_error(f"Błąd podczas generowania odpowiedzi po tool call: {e}")
            response_text = "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."

        self._history.append({"role": "model", "parts": [{"text": response_text}]})
        self._pending_tool_calls = None
        return OpenAIResponse(response_text)

    def _build_openai_messages(self) -> List[Dict[str, str]]:
        """
        Builds OpenAI messages format from universal history format.

        Returns:
            List of messages in OpenAI format: [{"role": "system|user|assistant", "content": "..."}]
        """
        openai_messages = []

        # Add system instruction as first message
        if self.system_instruction:
            openai_messages.append({
                "role": "system",
                "content": self.system_instruction
            })

        # Convert universal history format to OpenAI format
        for message in self._history:
            role = message["role"]
            text = message["parts"][0]["text"] if message["parts"] else ""

            # Convert role: "model" -> "assistant", "user" stays "user"
            openai_role = "assistant" if role == "model" else "user"

            openai_messages.append({
                "role": openai_role,
                "content": text
            })

        return openai_messages


class OpenAIResponse:
    """
    Response object that mimics the Gemini response interface.
    Provides a .text attribute containing the response text.
    """

    def __init__(self, text: str):
        self.text = text


class OpenAIResponseWithToolCalls:
    """
    Response wrapper for tool calls from OpenAI.
    """

    def __init__(self, tool_calls: List[Dict]):
        """
        Initialize with tool calls.

        Args:
            tool_calls: List of tool call dictionaries
        """
        self.text = None  # No text when tool call
        self.tool_calls = tool_calls

    def has_tool_calls(self) -> bool:
        """
        Check if response contains tool calls.

        Returns:
            True if tool calls present, False otherwise
        """
        return len(self.tool_calls) > 0


class OpenAIClient:
    """
    Encapsulates all OpenAI API interactions.
    Provides a clean interface compatible with GeminiLLMClient and LlamaClient.
    Compatible with OpenAI, Ollama, llama-server, and other OpenAI-compatible services.
    """

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None,
                 temperature: float = 0.7, top_p: float = 1.0):
        """
        Initialize the OpenAI client with explicit parameters.

        Args:
            model_name: Model to use (e.g., 'gpt-4', 'gpt-3.5-turbo', or Ollama model name)
            api_key: OpenAI API key (or dummy key for local services like Ollama)
            base_url: Optional base URL for OpenAI-compatible services (e.g., 'http://localhost:11434/v1' for Ollama)
            temperature: Controls randomness (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)

        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError("API key cannot be empty or None")

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p

        # Initialize the client during construction
        self._client = self._initialize_client()

    @staticmethod
    def preparing_for_use_message() -> str:
        """
        Returns a message indicating that OpenAI client is being prepared.

        Returns:
            Formatted preparation message string
        """
        return "🤖 Przygotowywanie klienta OpenAI..."

    @classmethod
    def from_environment(cls) -> 'OpenAIClient':
        """
        Factory method that creates an OpenAIClient instance from environment variables.

        Returns:
            OpenAIClient instance initialized with environment variables

        Raises:
            ValueError: If required environment variables are not set
        """
        load_dotenv()

        # Walidacja z Pydantic
        config = OpenAIConfig(
            model_name=os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo'),
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            openai_base_url=os.getenv('OPENAI_BASE_URL'),
            temperature=float(os.getenv('TEMPERATURE', 0.7)),
            top_p=float(os.getenv('TOP_P', 1.0))
        )

        if config.openai_base_url:
            console.print_info(f"Konfiguracja klienta OpenAI z bazowym URL: {config.openai_base_url}")

        return cls(
            model_name=config.model_name,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            temperature=config.temperature,
            top_p=config.top_p
        )

    def _initialize_client(self) -> OpenAI:
        """
        Initializes the OpenAI client.

        Returns:
            Initialized OpenAI client instance

        Raises:
            RuntimeError: If client initialization fails
        """
        try:
            console.print_info(f"Inicjalizacja klienta OpenAI: {self.model_name}")
            if self.base_url:
                console.print_info(f"Bazowy URL: {self.base_url}")
                return OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                return OpenAI(api_key=self.api_key)
        except Exception as e:
            console.print_error(f"Błąd inicjalizacji klienta OpenAI: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def create_chat_session(self,
                          system_instruction: str,
                          history: Optional[List[Dict]] = None,
                          thinking_budget: int = 0,
                          tools: Optional[List] = None) -> OpenAIChatSession:
        """
        Creates a new chat session with the specified configuration.

        Args:
            system_instruction: System role/prompt for the assistant
            history: Previous conversation history (optional, in universal dict format)
            thinking_budget: Ignored for OpenAI (compatibility parameter)
            tools: Optional list of ToolDefinition objects for function calling

        Returns:
            OpenAIChatSession object
        """
        if not self._client:
            raise RuntimeError("OpenAI client not initialized")

        return OpenAIChatSession(
            openai_client=self._client,
            model_name=self.model_name,
            system_instruction=system_instruction,
            history=history or [],
            temperature=self.temperature,
            top_p=self.top_p,
            tools=tools
        )

    def count_history_tokens(self, history: List[Dict]) -> int:
        """
        Counts tokens for the given conversation history.
        Note: This is an approximation using a simple heuristic.
        For accurate counting, consider using tiktoken library.

        Args:
            history: Conversation history in universal dict format

        Returns:
            Estimated token count
        """
        if not history:
            return 0

        try:
            # Simple approximation: ~4 characters per token
            # For production use, consider using tiktoken library for accurate counting
            text_parts = []
            for message in history:
                if "parts" in message and message["parts"]:
                    text_parts.append(message["parts"][0]["text"])

            full_text = " ".join(text_parts)
            # Rough estimation: 4 chars per token average
            return len(full_text) // 4

        except Exception as e:
            console.print_error(f"Błąd podczas liczenia tokenów: {e}")
            return 0

    def get_model_name(self) -> str:
        """Returns the currently configured model name."""
        return self.model_name

    def is_available(self) -> bool:
        """
        Checks if the LLM service is available and properly configured.

        Returns:
            True if client is properly initialized and has API key
        """
        return self._client is not None and bool(self.api_key)

    def ready_for_use_message(self) -> str:
        """
        Returns a ready-to-use message with model info and masked API key.

        Returns:
            Formatted message string for display
        """
        # Mask API key - show first 4 and last 4 characters
        if len(self.api_key) <= 8:
            masked_key = "****"
        else:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"

        base_msg = f"✅ Klient OpenAI gotowy do użycia (Model: {self.model_name}, Key: {masked_key}"

        if self.base_url:
            return f"{base_msg}, Base URL: {self.base_url})"
        else:
            return f"{base_msg})"

    @property
    def client(self):
        """
        Provides access to the underlying OpenAI client for backwards compatibility.
        This property should be used sparingly and eventually removed.
        """
        return self._client
