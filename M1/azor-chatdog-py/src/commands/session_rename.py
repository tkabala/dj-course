"""
Session Rename Command
Handles the /session rename <title> command to set or change the session title.
"""

from cli import console


def rename_session_command(session, command_parts: list):
    """
    Handles /session rename <title> command.

    Args:
        session: Current ChatSession instance
        command_parts: Full command split by spaces ['/session', 'rename', ...title words...]
    """
    # command_parts = ['/session', 'rename', 'My', 'Thread', 'Title']
    if len(command_parts) < 3:
        console.print_error("Błąd: Użycie: /session rename <tytuł>")
        return

    # Join everything after 'rename' as the title
    title = ' '.join(command_parts[2:])

    if session.set_title(title):
        console.print_info(f"✓ Ustawiono tytuł sesji: {title}")
        # Save immediately to persist the title
        session.save_to_file()
    else:
        console.print_error("Błąd: Tytuł nie może być pusty.")
