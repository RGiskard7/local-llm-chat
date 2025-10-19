"""
Local LLM Chat - Interfaz Universal para Modelos de Lenguaje Locales
Soporta modelos GGUF y Transformers con adaptación automática de system prompt
"""

__version__ = "1.0.0"
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
)

__all__ = [
    "UniversalChatClient",
    "detect_model_type",
    "get_chat_format",
    "supports_native_system",
    "get_model_info",
    "list_popular_models",
    "get_hardware_info",
]
