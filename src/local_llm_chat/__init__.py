"""
Local LLM Chat - Interfaz Universal para Modelos de Lenguaje Locales
Soporta modelos GGUF (llama.cpp) y Transformers (Hugging Face) con adaptación automática
"""

__version__ = "2.0.1"
__author__ = "Edu Díaz (RGiskard7)"
__license__ = "MIT"

from .client import UniversalChatClient
from .model_config import (
    detect_model_type,
    get_chat_format,
    supports_native_system,
    get_model_info,
    list_popular_models,
    get_hardware_info,
    detect_backend_type,
    is_gguf_model,
    is_transformers_model,
)
from .config import Config, RAGConfig, LLMConfig
from .backends import GGUFBackend, TransformersBackend, TRANSFORMERS_AVAILABLE

__all__ = [
    "UniversalChatClient",
    "detect_model_type",
    "get_chat_format",
    "supports_native_system",
    "get_model_info",
    "list_popular_models",
    "get_hardware_info",
    "detect_backend_type",
    "is_gguf_model",
    "is_transformers_model",
    "Config",
    "RAGConfig",
    "LLMConfig",
    "GGUFBackend",
    "TransformersBackend",
    "TRANSFORMERS_AVAILABLE",
]
