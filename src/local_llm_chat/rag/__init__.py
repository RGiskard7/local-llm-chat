"""
RAG Module - Sistema de Retrieval-Augmented Generation

Este módulo proporciona backends RAG modulares para enriquecer
las respuestas del LLM con contexto de documentos.

Arquitectura:
- RAGBackend: Interfaz abstracta
- SimpleRAG: Backend rápido con vectorstore
- RAGAnythingBackend: Backend complejo con knowledge graph
- RAGManager: Orquestador que permite cambiar backends fácilmente

Uso básico:
    from rag import RAGManager
    
    # Inicializar con SimpleRAG (recomendado para CPU)
    rag = RAGManager(client, backend="simple")
    
    # Cargar documento
    rag.load_document("/path/to/document.pdf")
    
    # Buscar contexto (sin LLM)
    result = rag.search_context("¿Qué dice sobre X?")
    
    # El LLM se usa externamente con el contexto
    response = client.infer(prompt_with_context)
"""

from .base import RAGBackend
from .manager import RAGManager

# Imports condicionales para backends RAG opcionales
# (mismo patrón que backends/__init__.py)
try:
    from .simple_rag_backend import SimpleRAG
except ImportError:
    SimpleRAG = None

try:
    from .raganything_backend import RAGAnythingBackend
except ImportError:
    RAGAnythingBackend = None

__all__ = [
    "RAGBackend",
    "RAGManager",
    "SimpleRAG",
    "RAGAnythingBackend",
]

__version__ = "2.0.0"

