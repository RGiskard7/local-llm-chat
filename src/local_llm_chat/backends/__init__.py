"""
Model Backends - Sistema de backends modulares para diferentes tipos de modelos

Este módulo proporciona una interfaz común para diferentes backends de modelos:
- GGUFBackend: Modelos GGUF vía llama-cpp-python
- TransformersBackend: Modelos Hugging Face Transformers

Arquitectura:
- ModelBackend: Interfaz abstracta común
- GGUFBackend: Backend para modelos GGUF locales
- TransformersBackend: Backend para modelos HF (local o remoto)

Uso básico:
    from backends import GGUFBackend, TransformersBackend
    
    # GGUF backend
    backend = GGUFBackend(model_path="model.gguf")
    
    # Transformers backend
    backend = TransformersBackend(model_name="bigscience/bloom")
    
    # Uso común
    response = backend.generate(messages, max_tokens=256)
"""

from .base import ModelBackend
from .gguf_backend import GGUFBackend

# Importación condicional de TransformersBackend
try:
    from .transformers_backend import TransformersBackend
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TransformersBackend = None
    TRANSFORMERS_AVAILABLE = False

__all__ = [
    "ModelBackend",
    "GGUFBackend",
    "TransformersBackend",
    "TRANSFORMERS_AVAILABLE",
]

__version__ = "1.0.0"

