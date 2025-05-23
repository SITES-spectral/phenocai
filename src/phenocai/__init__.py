"""PhenoCAI - Phenological Camera AI Analysis Package"""

__version__ = "0.1.0"

# Make CLI available at package level
from .cli import cli

__all__ = ['cli', '__version__']