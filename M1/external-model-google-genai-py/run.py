from google import genai
from google.genai import types as genai_types
import os
from transformers import AutoTokenizer
from colorama import Fore, Style, init

from dotenv import load_dotenv
load_dotenv()

init() # Initialize Colorama for cross-platform compatibility

# if set, print first 4 chars and last 4 chars and dots inside, else print NOT SET
print(f"env var \"GEMINI_API_KEY\" is: { os.getenv('GEMINI_API_KEY', '')[:4] + '...' + os.getenv('GEMINI_API_KEY', '')[-4:] if len(os.getenv('GEMINI_API_KEY', '')) > 0 else 'NOT SET' }")
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to your Google Gemini API key.")

client = genai.Client()

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_ROLE = "you were Gandalf the Grey in the Lord of the Rings. You answer in max 15 words."

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define colors
USER_COLOR = Fore.CYAN
GANDALF_COLOR = Fore.MAGENTA
TOKEN_COLOR = Fore.YELLOW
RESET_COLOR = Style.RESET_ALL

def main():
    """Main function to run the interactive chat."""
    conversation_history: list[genai_types.Content] = []
    print("--- You are now chatting with Gandalf. Type 'exit' or 'quit' to end the conversation. ---")

    while True:
        try:
            user_input = input(f"\n{USER_COLOR}You: {RESET_COLOR}")

            if user_input.lower() in ['exit', 'quit']:
                print(f"\n{GANDALF_COLOR}Gandalf: Farewell, until our paths cross again.{RESET_COLOR}")
                break

            user_content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_input)])
            user_tokens_list = tokenizer.encode(user_input)
            print(f"{TOKEN_COLOR}(Tokens: {user_tokens_list}){RESET_COLOR}")

            conversation_history.append(user_content)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=conversation_history,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_ROLE,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0) # Disables thinking
                ),
            )

            model_response_text = response.text
            model_content = genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=model_response_text)])
            model_tokens_list = tokenizer.encode(model_response_text)

            print(f"\n{GANDALF_COLOR}Gandalf: {model_response_text}{RESET_COLOR}")
            print(f"{TOKEN_COLOR}(Tokens: {model_tokens_list}){RESET_COLOR}")

            conversation_history.append(model_content)

        except (KeyboardInterrupt, EOFError):
            print(f"\n{GANDALF_COLOR}Gandalf: The journey is interrupted. Farewell for now.{RESET_COLOR}")
            break

if __name__ == "__main__":
    main()