"""
Google Gemini LLM Client Implementation
Encapsulates all Google Gemini AI interactions.
"""

import os
import sys
from typing import Optional, List, Any, Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv
from cli import console
from .gemini_validation import GeminiConfig

class GeminiResponseWrapper:
    """
    Wrapper for Gemini response that provides tool call detection.
    """

    def __init__(self, gemini_response):
        """
        Initialize wrapper with Gemini response.

        Args:
            gemini_response: The actual Gemini response object
        """
        self.raw_response = gemini_response
        # First extract tool calls to check if there are any
        self.tool_calls = self._extract_tool_calls()
        # Only try to get text if there are NO tool calls (avoids warning)
        if len(self.tool_calls) == 0:
            self.text = gemini_response.text if hasattr(gemini_response, 'text') else None
        else:
            self.text = None  # No text when tool call is present

    def _extract_tool_calls(self) -> List[Dict]:
        """
        Extract tool calls from Gemini response.

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []

        # Check if response has candidates with function calls
        if hasattr(self.raw_response, 'candidates') and self.raw_response.candidates:
            for candidate in self.raw_response.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            tool_calls.append({
                                "name": fc.name,
                                "arguments": dict(fc.args) if hasattr(fc, 'args') else {}
                            })

        return tool_calls

    def has_tool_calls(self) -> bool:
        """
        Check if response contains tool calls.

        Returns:
            True if tool calls present, False otherwise
        """
        return len(self.tool_calls) > 0


class GeminiChatSessionWrapper:
    """
    Wrapper for Gemini chat session that provides universal dictionary-based history format.
    This ensures compatibility with LlamaClient's history format.
    """
    
    def __init__(self, gemini_session):
        """
        Initialize wrapper with Gemini chat session.
        
        Args:
            gemini_session: The actual Gemini chat session object
        """
        self.gemini_session = gemini_session
    
    def send_message(self, text: str) -> Any:
        """
        Forwards message to Gemini session.

        Args:
            text: User's message

        Returns:
            Response object from Gemini with tool call handling
        """
        response = self.gemini_session.send_message(text)

        # Wrap response to add tool call detection
        return GeminiResponseWrapper(response)

    def send_tool_results(self, tool_results: list) -> Any:
        """
        Send function response parts back to the model after tool execution.

        Args:
            tool_results: list of {"name": str, "result": dict}

        Returns:
            Response object from Gemini
        """
        parts = []
        for tr in tool_results:
            result_content = tr["result"].get("message", str(tr["result"]))
            parts.append(types.Part.from_function_response(
                name=tr["name"],
                response={"output": result_content}
            ))
        response = self.gemini_session.send_message(parts)
        return GeminiResponseWrapper(response)
    
    def get_history(self) -> List[Dict]:
        """
        Gets conversation history in universal dictionary format.
        
        Returns:
            List of dictionaries with format: {"role": "user|model", "parts": [{"text": "..."}]}
        """
        gemini_history = self.gemini_session.get_history()
        universal_history = []
        
        for content in gemini_history:
            # Convert Gemini Content object to universal dictionary format
            text_part = ""
            if hasattr(content, 'parts') and content.parts:
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_part = part.text
                        break
            
            if text_part:
                universal_content = {
                    "role": content.role,
                    "parts": [{"text": text_part}]
                }
                universal_history.append(universal_content)
        
        return universal_history

class GeminiLLMClient:
    """
    Encapsulates all Google Gemini AI interactions.
    Provides a clean interface for chat sessions, token counting, and configuration.
    """
    
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7,
                 top_p: float = 1.0, top_k: int = 40):
        """
        Initialize the Gemini LLM client with explicit parameters.

        Args:
            model_name: Model to use (e.g., 'gemini-2.5-flash')
            api_key: Google Gemini API key
            temperature: Controls randomness (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Number of top tokens to sample from

        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError("API key cannot be empty or None")

        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # Initialize the client during construction
        self._client = self._initialize_client()
    
    @staticmethod
    def preparing_for_use_message() -> str:
        """
        Returns a message indicating that Gemini client is being prepared.
        
        Returns:
            Formatted preparation message string
        """
        return "🤖 Przygotowywanie klienta Gemini..."
    
    @classmethod
    def from_environment(cls) -> 'GeminiLLMClient':
        """
        Factory method that creates a GeminiLLMClient instance from environment variables.

        Returns:
            GeminiLLMClient instance initialized with environment variables

        Raises:
            ValueError: If required environment variables are not set
        """
        load_dotenv()

        # Walidacja z Pydantic
        config = GeminiConfig(
            model_name=os.getenv('MODEL_NAME', 'gemini-2.5-flash'),
            gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
            temperature=float(os.getenv('TEMPERATURE', 0.7)),
            top_p=float(os.getenv('TOP_P', 1.0)),
            top_k=int(os.getenv('TOP_K', 40))
        )

        return cls(
            model_name=config.model_name,
            api_key=config.gemini_api_key,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        )
    
    def _initialize_client(self) -> genai.Client:
        """
        Initializes the Google GenAI client.
        
        Returns:
            Initialized GenAI client
            
        Raises:
            SystemExit: If client initialization fails
        """
        try:
            return genai.Client()
        except Exception as e:
            console.print_error(f"Błąd inicjalizacji klienta Gemini: {e}")
            sys.exit(1)
    
    def create_chat_session(self,
                          system_instruction: str,
                          history: Optional[List[Dict]] = None,
                          thinking_budget: int = 0,
                          tools: Optional[List] = None) -> GeminiChatSessionWrapper:
        """
        Creates a new chat session with the specified configuration.

        Args:
            system_instruction: System role/prompt for the assistant
            history: Previous conversation history (optional, in universal dict format)
            thinking_budget: Thinking budget for the model
            tools: Optional list of ToolDefinition objects for function calling

        Returns:
            GeminiChatSessionWrapper with universal dictionary-based interface
        """
        if not self._client:
            raise RuntimeError("LLM client not initialized")
        
        # Convert universal dict format to Gemini Content objects
        gemini_history = []
        if history:
            for entry in history:
                if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
                    text = entry['parts'][0].get('text', '') if entry['parts'] else ''
                    if text:
                        content = types.Content(
                            role=entry['role'],
                            parts=[types.Part.from_text(text=text)]
                        )
                        gemini_history.append(content)

        # Convert tools to Gemini format if provided
        gemini_tools = None
        if tools:
            gemini_tools = [tool.to_gemini_tool() for tool in tools]

        # Build config parameters
        config_params = {
            "system_instruction": system_instruction,
            "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

        # Add tools if provided
        if gemini_tools:
            config_params["tools"] = gemini_tools

        gemini_session = self._client.chats.create(
            model=self.model_name,
            history=gemini_history,
            config=types.GenerateContentConfig(**config_params)
        )
        
        return GeminiChatSessionWrapper(gemini_session)
    
    def count_history_tokens(self, history: List[Dict]) -> int:
        """
        Counts tokens for the given conversation history.
        
        Args:
            history: Conversation history in universal dict format
            
        Returns:
            Total token count
        """
        if not history:
            return 0
        
        try:
            # Convert universal dict format to Gemini Content objects for token counting
            gemini_history = []
            for entry in history:
                if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
                    text = entry['parts'][0].get('text', '') if entry['parts'] else ''
                    if text:
                        content = types.Content(
                            role=entry['role'],
                            parts=[types.Part.from_text(text=text)]
                        )
                        gemini_history.append(content)
            
            response = self._client.models.count_tokens(
                model=self.model_name,
                contents=gemini_history
            )
            return response.total_tokens
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
        
        return f"✅ Klient Gemini gotowy do użycia (Model: {self.model_name}, Key: {masked_key})"
    
    @property
    def client(self):
        """
        Provides access to the underlying GenAI client for backwards compatibility.
        This property should be used sparingly and eventually removed.
        """
        return self._client
