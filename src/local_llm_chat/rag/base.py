"""
RAG Backend - Interfaz abstracta

Define el contrato que todos los backends RAG deben cumplir.
Arquitectura: RAG solo maneja indexación y búsqueda de contexto.
El LLM Client maneja la generación de respuestas externamente.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class RAGBackend(ABC):
    """
    Interfaz abstracta para backends RAG
    
    Responsabilidades:
    - Indexar documentos (load_document)
    - Gestionar múltiples documentos (unload_document, list_documents, clear_all)
    - Buscar contexto relevante (search_context)
    - Reportar estado (get_status)
    
    NO incluye:
    - Generación de respuestas (responsabilidad del LLM Client)
    - Formateo de prompts (responsabilidad del Router/CLI)
    """
    
    @abstractmethod
    def load_document(self, file_path: str) -> bool:
        """
        Carga e indexa un documento en el sistema RAG
        
        Args:
            file_path: Ruta al documento a procesar
            
        Returns:
            True si exitoso, False si falla
        """
        pass
    
    @abstractmethod
    def unload_document(self, file_path: str) -> bool:
        """
        Remueve un documento del índice RAG
        
        Args:
            file_path: Ruta al documento a remover
            
        Returns:
            True si exitoso, False si no se encuentra
        """
        pass
    
    @abstractmethod
    def list_documents(self) -> List[str]:
        """
        Lista todos los documentos cargados en el sistema
        
        Returns:
            Lista de rutas de documentos cargados
        """
        pass
    
    @abstractmethod
    def clear_all_documents(self) -> bool:
        """
        Limpia todos los documentos del índice RAG
        
        Returns:
            True si exitoso
        """
        pass
    
    @abstractmethod
    def search_context(self, question: str, **kwargs) -> Dict:
        """
        Busca contexto relevante para una pregunta (sin llamar al LLM)
        
        Args:
            question: Pregunta del usuario
            **kwargs: Parámetros adicionales (top_k, filters, etc.)
            
        Returns:
            dict con estructura:
            {
                "contexts": List[str],          # Chunks de texto relevantes
                "sources": List[dict],          # Metadatos de fuentes
                "relevance_scores": List[float] # Scores de similitud
            }
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict:
        """
        Retorna el estado actual del RAG
        
        Returns:
            dict con información del backend (documentos, estadísticas, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def current_document(self) -> Optional[str]:
        """Documento actual cargado en el sistema (primer documento si hay múltiples)"""
        pass

