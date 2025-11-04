"""
RAG Manager - Orquestador de backends RAG

Permite cambiar fácilmente entre diferentes backends RAG manteniendo
una interfaz uniforme.
"""
from typing import Dict, List, Optional

from .base import RAGBackend
from ..config import Config


class RAGManager:
    """
    Gestor principal de RAG - Arquitectura modular profesional
    
    Permite cambiar entre diferentes backends RAG fácilmente:
    - "simple": SimpleRAG (rápido, sin knowledge graph)
    - "raganything": RAG-Anything (complejo, con knowledge graph)
    
    Características profesionales:
    - Gestión de múltiples documentos
    - Modo RAG activable/desactivable (rag_mode)
    - Soporte para /load, /unload, /list, /clear
    - Interfaz uniforme para todos los backends
    
    Uso:
        # SimpleRAG (recomendado para CPU)
        rag = RAGManager(client, backend="simple")
        rag.load_document("doc1.pdf")
        rag.rag_mode = True  # Activar RAG
        
        # RAG-Anything (recomendado para GPU)
        rag = RAGManager(client, backend="raganything")
    """
    
    AVAILABLE_BACKENDS = {
        "simple": None,  # Se carga dinámicamente
        "raganything": None,  # Se carga dinámicamente
    }
    
    def __init__(self, client, backend: str = "simple", working_dir: Optional[str] = None, config: Optional[Config] = None):
        """
        Inicializa el RAGManager con el backend seleccionado
        
        Args:
            client: Cliente LLM (solo necesario para raganything)
            backend: Nombre del backend ("simple" o "raganything")
            working_dir: Directorio de trabajo (None = usar default del backend)
            config: Configuración opcional (usa default si None)
        """
        self.client = client
        self.backend_name = backend
        self.rag_mode = False  # RAG desactivado por defecto
        self.config = config or Config()  # Configuración
        
        if backend not in self.AVAILABLE_BACKENDS:
            raise ValueError(
                f"Backend '{backend}' no disponible. "
                f"Opciones: {list(self.AVAILABLE_BACKENDS.keys())}"
            )
        
        # Cargar backend dinámicamente
        if backend == "simple":
            from .simple_rag_backend import SimpleRAG
            backend_class = SimpleRAG
            if working_dir is None:
                working_dir = "./simple_rag_data"
        elif backend == "raganything":
            from .raganything_backend import RAGAnythingBackend
            backend_class = RAGAnythingBackend
            if working_dir is None:
                working_dir = "./rag_data"
        
        # Inicializar el backend seleccionado (pasando configuración)
        self.backend: RAGBackend = backend_class(client, working_dir, config=self.config)
    
    def load_document(self, file_path: str) -> bool:
        """
        Carga documento usando el backend activo
        
        Args:
            file_path: Ruta al documento
            
        Returns:
            True si exitoso
        """
        success = self.backend.load_document(file_path)
        if success and not self.rag_mode:
            print("[RAG] Document loaded. Use '/rag on' to activate RAG mode")
        return success
    
    def unload_document(self, file_path: str) -> bool:
        """
        Remueve un documento del índice
        
        Args:
            file_path: Ruta al documento a remover
            
        Returns:
            True si exitoso
        """
        return self.backend.unload_document(file_path)
    
    def list_documents(self) -> List[str]:
        """
        Lista todos los documentos cargados
        
        Returns:
            Lista de rutas de documentos
        """
        return self.backend.list_documents()
    
    def clear_all_documents(self) -> bool:
        """
        Limpia todos los documentos del índice
        
        Returns:
            True si exitoso
        """
        success = self.backend.clear_all_documents()
        if success:
            self.rag_mode = False  # Desactivar RAG al limpiar
        return success
    
    def search_context(self, question: str, **kwargs) -> Dict:
        """
        Busca contexto relevante sin llamar al LLM (arquitectura desacoplada)
        
        Args:
            question: Pregunta del usuario
            **kwargs: Parámetros adicionales para el backend (ej. top_k)
            
        Returns:
            dict con contexts, sources, relevance_scores
        """
        return self.backend.search_context(question, **kwargs)
    
    def query(self, question: str) -> str:
        """
        Query completa usando el backend activo (LEGACY para RAG-Anything)
        
        NOTA: Este método está deprecated para SimpleRAG.
        Usar search_context() + LLM externo en su lugar.
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Respuesta del RAG (solo para backends con LLM integrado)
        """
        if hasattr(self.backend, 'query'):
            return self.backend.query(question)
        else:
            raise NotImplementedError(
                f"Backend '{self.backend_name}' no soporta query() directo. "
                f"Usa search_context() + LLM externo."
            )
    
    def get_status(self) -> Dict:
        """Estado del sistema RAG"""
        status = self.backend.get_status()
        status["manager_backend"] = self.backend_name
        status["rag_mode"] = self.rag_mode
        return status
    
    @property
    def current_document(self) -> Optional[str]:
        """Documento actual cargado (primer documento si hay múltiples)"""
        return self.backend.current_document

