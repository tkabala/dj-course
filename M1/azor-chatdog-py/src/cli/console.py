"""
Console output utilities for the chatbot.
Centralizes colorama/rich usage for consistent terminal output.
"""
import sys
from colorama import init, Fore, Style
from rich.console import Console
from rich.markdown import Markdown
from files.config import LOG_DIR

init(autoreset=True)

_rich_console = Console()


def print_error(message: str):
    """Print an error message in red color."""
    print(Fore.RED + message + Style.RESET_ALL)


def print_assistant(message: str):
    """Print an assistant-role message in cyan color (used for session display etc.)."""
    print(Fore.CYAN + message + Style.RESET_ALL)


def print_assistant_response(name: str, response: str):
    """Print assistant name in bold magenta and render the response as markdown.

    Args:
        name: The assistant name (e.g. "AZOR")
        response: The markdown response text
    """
    _rich_console.print(f"\n[bold magenta]{name}:[/bold magenta]")
    _rich_console.print(Markdown(response))


def print_user(message: str):
    """Print a user message in blue color."""
    print(Fore.BLUE + message + Style.RESET_ALL)


def print_info(message: str):
    """Print an informational message."""
    print(message)


def print_help(message: str):
    """Print a help message in yellow color."""
    print(Fore.YELLOW + message + Style.RESET_ALL)


def display_help(session_id: str):
    """Displays a short help message."""
    print_info(f"Aktualna sesja (ID): {session_id}")
    print_info(f"Pliki sesji są zapisywane na bieżąco w: {LOG_DIR}")
    print_help("Dostępne komendy (slash commands):")
    print_help("  /switch <ID>      - Przełącza na istniejącą sesję.")
    print_help("  /help             - Wyświetla tę pomoc.")
    print_help("  /exit, /quit      - Zakończenie czatu.")
    print_help("\n  /session list     - Wyświetla listę dostępnych sesji.")
    print_help("  /session display  - Wyświetla całą historię sesji.")
    print_help("  /session title    - Wyświetla tytuł bieżącej sesji.")
    print_help("  /session rename <tytuł> - Ustawia/zmienia tytuł sesji.")
    print_help("  /session pop      - Usuwa ostatnią parę wpisów (TY i asystent).")
    print_help("  /session clear    - Czyści historię bieżącej sesji.")
    print_help("  /session new      - Rozpoczyna nową sesję.")
    print_help("  /session remove   - Usuwa wybraną sesję.")
    print_help("\n  /pdf              - Eksportuje sesję do PDF.")
    print_help("  /audio            - Generuje plik WAV z ostatnią odpowiedzią Azora.")
    print_help("  /audio-all        - Generuje plik WAV z całą konwersacją.")
    print_help("\n  /role             - Wyświetla aktualną rolę i dostępne role.")
    print_help("  /role <nazwa>     - Zmienia asystenta (azor / reksio). Historia zostaje.")


def display_final_instructions(session_id: str):
    """Displays instructions for continuing the session."""
    print_info("\n--- Instrukcja Kontynuacji Sesji ---")
    print_info(f"Aby kontynuować tę sesję (ID: {session_id}) później, użyj komendy:")
    print(Fore.WHITE + Style.BRIGHT + f"\n    python {sys.argv[0]} --session-id={session_id}\n" + Style.RESET_ALL)
    print("--------------------------------------\n")


def show_selection_list(question: str, options: list) -> str | None:
    """Display an inline keyboard-navigated list and return the chosen option, or None if cancelled."""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    selected_index = [0]
    result = [None]

    kb = KeyBindings()

    @kb.add('up')
    def move_up(event):
        selected_index[0] = (selected_index[0] - 1) % len(options)
        event.app.invalidate()

    @kb.add('down')
    def move_down(event):
        selected_index[0] = (selected_index[0] + 1) % len(options)
        event.app.invalidate()

    @kb.add('enter')
    def confirm(event):
        result[0] = options[selected_index[0]]
        event.app.exit()

    @kb.add('escape')
    @kb.add('c-c')
    def cancel(event):
        event.app.exit()

    def get_formatted_text():
        tokens = [('class:question', f'{question}\n')]
        for i, opt in enumerate(options):
            if i == selected_index[0]:
                tokens.append(('class:selected', f'  ▶ {opt}\n'))
            else:
                tokens.append(('', f'    {opt}\n'))
        tokens.append(('class:hint', '  [↑↓] nawigacja  [Enter] wybierz  [Esc] anuluj'))
        return tokens

    from prompt_toolkit.styles import Style
    style = Style.from_dict({
        'question': 'bold ansiyellow',
        'selected': 'bold ansicyan',
        'hint':     'ansibrightblack',
    })

    layout = Layout(Window(FormattedTextControl(get_formatted_text)))
    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        erase_when_done=True,
    )
    app.run()
    return result[0]
