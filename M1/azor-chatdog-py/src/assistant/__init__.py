"""
Assistant module initialization
Exports the Assistant class and assistant factory functions.
"""

from .assistent import Assistant
from .azor import create_azor_assistant
from .reksio import create_reksio_assistant
from .angel_investor import create_angel_investor_assistant
from .sparring_partner import create_sparring_partner_assistant

ASSISTANT_REGISTRY = {
    'azor': create_azor_assistant,
    'reksio': create_reksio_assistant,
    'angel-investor': create_angel_investor_assistant,
    'sparring-partner': create_sparring_partner_assistant,
}

__all__ = ['Assistant', 'create_azor_assistant', 'create_reksio_assistant',
           'create_angel_investor_assistant', 'create_sparring_partner_assistant',
           'ASSISTANT_REGISTRY']
