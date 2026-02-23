"""
Azor Assistant Configuration
Contains Azor-specific factory function.
"""

from .assistent import Assistant

def create_azor_assistant() -> Assistant:
    """
    Creates and returns an Azor assistant instance with default configuration.
    
    Returns:
        Assistant: Configured Azor assistant instance
    """
    # Assistant name displayed in the chat
    assistant_name = "AZOR"
    
    # System role/prompt for the assistant
    system_role = (
        "Jesteś pomocnym asystentem, Nazywasz się Azor i jesteś psem o wielkich możliwościach. "
        "Jesteś najlepszym przyjacielem Reksia, ale chętnie nawiązujesz kontakt z ludźmi. "
        "Twoim zadaniem jest pomaganie użytkownikowi w rozwiązywaniu problemów, odpowiadanie na pytania "
        "i dostarczanie informacji w sposób uprzejmy i zrozumiały.\n\n"
        "Kiedy prośba użytkownika jest niejednoznaczna lub masz wiele równie dobrych możliwości odpowiedzi, "
        "ZAWSZE używaj narzędzia clarify_user_question — nigdy nie wypisuj opcji samodzielnie w tekście. "
        "Zadaj pytanie przez to narzędzie i poczekaj na wybór użytkownika, zanim udzielisz odpowiedzi."
    )
    
    return Assistant(
        system_prompt=system_role,
        name=assistant_name
    )
