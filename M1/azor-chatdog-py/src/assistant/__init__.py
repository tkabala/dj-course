"""
Assistant module initialization
Exports the Assistant class and assistant factory functions.
"""

from .assistent import Assistant
from .azor import create_azor_assistant
from .reksio import create_reksio_assistant

ASSISTANT_REGISTRY = {
    'azor': create_azor_assistant,
    'reksio': create_reksio_assistant,
}

__all__ = ['Assistant', 'create_azor_assistant', 'create_reksio_assistant', 'ASSISTANT_REGISTRY']
