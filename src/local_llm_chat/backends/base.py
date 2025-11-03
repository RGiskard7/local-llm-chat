"""
Model Backend - Interfaz abstracta común para todos los backends de modelos

Define el contrato que todos los backends deben cumplir para garantizar
intercambiabilidad total entre GGUF, Transformers y futuros backends.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class ModelBackend(ABC):
    """
    Interfaz abstracta para backends de modelos
    
    Responsabilidades:
    - Cargar modelo (local o remoto)
    - Generar respuestas dado un historial de mensajes
    - Adaptar system prompts según capacidades del modelo
    - Reportar información del modelo y métricas
    - Descargar modelo si es necesario
    
    Todos los backends deben implementar esta interfaz para garantizar
    compatibilidad con UniversalChatClient, CLI y RAG.
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa el backend
        
        Args:
            **kwargs: Parámetros específicos del backend
        """
        self.model_type = "unknown"
        self.supports_native_system = False
        self.chat_format = None
        
    @abstractmethod
    def load_model(self) -> bool:
        """
        Carga el modelo en memoria
        
        Returns:
            True si exitoso, False si falla
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera respuesta dado un historial de mensajes
        
        Args:
            messages: Lista de mensajes [{"role": "user/assistant/system", "content": "..."}]
            max_tokens: Máximo de tokens a generar
            temperature: Temperatura para sampling
            **kwargs: Parámetros adicionales específicos del backend
            
        Returns:
            dict con estructura:
            {
                "content": str,                    # Texto de la respuesta
                "usage": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int
                },
                "elapsed_seconds": float
            }
        """
        pass
    
    @abstractmethod
    def unload_model(self):
        """
        Descarga el modelo de memoria
        
        Libera recursos (GPU/RAM) ocupados por el modelo
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del modelo cargado
        
        Returns:
            dict con información: nombre, tipo, tamaño, etc.
        """
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Indica si el modelo está cargado en memoria"""
        pass
    
    def format_messages(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Formatea mensajes adaptando el system prompt según capacidades del modelo
        
        Args:
            messages: Lista de mensajes
            system_prompt: System prompt opcional
            
        Returns:
            Mensajes formateados según el backend
        """
        formatted = []
        
        if system_prompt:
            if self.supports_native_system:
                # Soporte nativo: agregar como rol system
                formatted.append({"role": "system", "content": system_prompt})
            else:
                # Workaround: agregar como user + assistant
                formatted.append({"role": "user", "content": system_prompt})
                formatted.append({"role": "assistant", "content": "Understood. I will follow these instructions."})
        
        # Agregar resto de mensajes
        formatted.extend(messages)
        
        return formatted
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estima el número de tokens en un texto
        
        Args:
            text: Texto a estimar
            
        Returns:
            Número estimado de tokens (aproximación simple: palabras * 1.3)
        """
        # Aproximación simple: 1 token ≈ 0.75 palabras
        words = len(text.split())
        return int(words * 1.3)

