"""
Roleplay command: simulates an autonomous conversation between 2+ personas.
"""

from cli import console
from assistant import ASSISTANT_REGISTRY


def handle_roleplay_command(parts, manager):
    current = manager.get_current_session()

    # Guard: roleplay requires Gemini engine
    from llm.gemini_client import GeminiLLMClient
    if not isinstance(current._llm_client, GeminiLLMClient):
        console.print_error("Roleplay wymaga silnika Gemini. Zmień klienta i spróbuj ponownie.")
        return

    llm_client = current._llm_client

    # Persona selection
    available = list(ASSISTANT_REGISTRY.keys())
    selected_names = []

    while True:
        options = [n for n in available if n not in selected_names]
        if len(selected_names) >= 2:
            options = ["[Gotowe - zacznij roleplay]"] + options
        if not options:
            break

        question = f"Wybierz personę ({len(selected_names)} wybrano): " + ", ".join(selected_names) if selected_names else "Wybierz pierwszą personę:"
        choice = console.show_selection_list(question, options)

        if choice is None:
            console.print_info("Roleplay anulowany.")
            return
        if choice == "[Gotowe - zacznij roleplay]":
            break
        selected_names.append(choice)

    if len(selected_names) < 2:
        console.print_error("Roleplay wymaga co najmniej 2 person.")
        return

    personas = [ASSISTANT_REGISTRY[name]() for name in selected_names]

    console.print_info(f"\nRoleplay z: {', '.join(p.name for p in personas)}")
    initial_prompt = input("Podaj temat/prompt startowy: ").strip()
    if not initial_prompt:
        console.print_error("Prompt nie może być pusty.")
        return

    conversation = []  # list of (persona_idx: int, text: str)

    # Round-robin conversation loop
    persona_idx = 0
    while True:
        persona = personas[persona_idx]
        history = _build_roleplay_history(initial_prompt, conversation, persona_idx)
        text = _generate_roleplay_response(llm_client, persona, history)
        if text is None:
            console.print_error(f"Błąd generowania odpowiedzi dla {persona.name}.")
            break

        conversation.append((persona_idx, text))
        console.print_assistant_response(persona.name, text)

        if not console.wait_for_continue():
            break

        persona_idx = (persona_idx + 1) % len(personas)


def _build_roleplay_history(initial_prompt: str, conversation: list, current_idx: int) -> list:
    history = [{"role": "user", "parts": [{"text": initial_prompt}]}]
    for persona_idx, text in conversation:
        role = "model" if persona_idx == current_idx else "user"
        history.append({"role": role, "parts": [{"text": text}]})
    return history


def _generate_roleplay_response(llm_client, persona, history: list) -> str | None:
    from google.genai import types

    contents = []
    for entry in history:
        text = entry["parts"][0]["text"]
        contents.append(
            types.Content(
                role=entry["role"],
                parts=[types.Part.from_text(text=text)]
            )
        )

    try:
        response = llm_client._client.models.generate_content(
            model=llm_client.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=persona.system_prompt,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text
    except Exception as e:
        from cli import console as _console
        _console.print_error(f"Błąd API: {e}")
        return None
