"""
models/__init__.py

Импортирует все модули с реализациями, чтобы декораторы @registry.register()
сработали до первого обращения к реестру.
"""

from . import toxicity, empathy, semantic  # noqa: F401
