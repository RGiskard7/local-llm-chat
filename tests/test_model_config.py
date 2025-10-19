"""
Pruebas básicas para el módulo model_config
"""

import pytest
from local_llm_chat.model_config import (
    detect_model_type,
    get_chat_format,
    supports_native_system,
    estimate_model_size,
)


def test_detect_llama_3():
    """Prueba la detección de modelos Llama 3"""
    assert detect_model_type("llama-3-8b-instruct.gguf") == "llama-3"
    assert detect_model_type("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf") == "llama-3"


def test_detect_gemma():
    """Prueba la detección de modelos Gemma"""
    assert detect_model_type("gemma-2-9b-it-Q8_0.gguf") == "gemma"


def test_detect_mistral():
    """Prueba la detección de modelos Mistral"""
    assert detect_model_type("mistral-7b-instruct-v0.2.gguf") == "mistral"


def test_get_chat_format():
    """Prueba el mapeo de formato de chat"""
    assert get_chat_format("llama-3") == "llama-3"
    assert get_chat_format("gemma") == "gemma"
    assert get_chat_format("mistral") == "mistral"


def test_native_system_support():
    """Prueba la detección de soporte de system prompt"""
    assert supports_native_system("llama-3") == False  # Not in MODELS_WITH_NATIVE_SYSTEM_SUPPORT
    assert supports_native_system("mistral") == False
    assert supports_native_system("gemma") == False


def test_estimate_model_size():
    """Prueba la estimación de tamaño de modelo"""
    size_8b_q8 = estimate_model_size("llama-3-8b-q8_0.gguf")
    size_8b_q4 = estimate_model_size("llama-3-8b-q4_0.gguf")

    assert size_8b_q8 > size_8b_q4  # Q8 should be larger than Q4
    assert 6.0 <= size_8b_q8 <= 10.0  # Reasonable range for 8B Q8
    assert 3.0 <= size_8b_q4 <= 6.0   # Reasonable range for 8B Q4


def test_unknown_model():
    """Prueba el manejo de modelos desconocidos"""
    assert detect_model_type("some-random-model.gguf") == "unknown"
    assert get_chat_format("unknown") is None
