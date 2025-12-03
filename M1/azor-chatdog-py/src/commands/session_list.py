from files import session_files
from cli import console

def list_sessions_command():
    """Displays a formatted list of available sessions."""
    sessions = session_files.list_sessions()
    if sessions:
        console.print_help("\n--- Dostępne zapisane sesje (ID) ---")
        for session in sessions:
            if session.get('error'):
                console.print_error(f"- ID: {session['id']} ({session['error']})")
            else:
                # Display title if present
                title_display = f" - {session['title']}" if session.get('title') else ""
                console.print_help(
                    f"- ID: {session['id']}{title_display} "
                    f"(Wiadomości: {session['messages_count']}, "
                    f"Ost. aktywność: {session['last_activity']})"
                )
        console.print_help("------------------------------------")
    else:
        console.print_help("\nBrak zapisanych sesji.")
