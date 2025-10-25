"""
Configuración centralizada para Local LLM Chat

Sistema de configuración híbrido:
- Valores por defecto en código
- Archivo config.json opcional
- Variables de entorno para override
- Parámetros en constructor para uso como librería
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RAGConfig:
    """
    Configuración para el sistema RAG
    
    Attributes:
        chunk_size: Tamaño del chunk en palabras (default: 150)
        chunk_overlap: Overlap entre chunks en palabras (default: 25)
        top_k: Número de chunks a recuperar (default: 1)
        max_context_tokens: Límite máximo de tokens en contexto (default: 800)
    """
    chunk_size: int = 150
    chunk_overlap: int = 25
    top_k: int = 1
    max_context_tokens: int = 800


@dataclass
class LLMConfig:
    """
    Configuración para el modelo LLM
    
    Attributes:
        max_tokens: Máximo de tokens a generar (default: 256)
        temperature: Temperatura para sampling (default: 0.1)
        top_p: Nucleus sampling threshold (default: 0.9)
        repeat_penalty: Penalización por repetición (default: 1.1)
    """
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    repeat_penalty: float = 1.1


class Config:
    """
    Clase principal de configuración
    
    Carga configuración desde múltiples fuentes con prioridad:
    1. Valores por defecto (hardcoded)
    2. Archivo config.json (si existe)
    3. Variables de entorno (override)
    4. Parámetros en constructor (máxima prioridad)
    
    Uso:
        # Valores por defecto
        config = Config()
        
        # Con archivo JSON
        config = Config(config_file="my_config.json")
        
        # Con overrides
        config = Config()
        config.rag.chunk_size = 200
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa configuración
        
        Args:
            config_file: Ruta a archivo JSON de configuración (opcional)
        """
        # 1. Valores por defecto
        self.rag = RAGConfig()
        self.llm = LLMConfig()
        
        # 2. Cargar desde archivo si existe
        if config_file:
            self.load_from_file(config_file)
        else:
            # Intentar cargar config.json por defecto
            default_config = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(default_config):
                self.load_from_file(default_config)
        
        # 3. Override con variables de entorno
        self._load_from_env()
    
    def load_from_file(self, config_file: str):
        """
        Carga configuración desde archivo JSON
        
        Args:
            config_file: Ruta al archivo JSON
        """
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                
                # Actualizar configuración RAG
                if 'rag' in data:
                    for key, value in data['rag'].items():
                        if hasattr(self.rag, key):
                            setattr(self.rag, key, value)
                
                # Actualizar configuración LLM
                if 'llm' in data:
                    for key, value in data['llm'].items():
                        if hasattr(self.llm, key):
                            setattr(self.llm, key, value)
        
        except Exception as e:
            print(f"[WARNING] Could not load config from {config_file}: {e}")
    
    def _load_from_env(self):
        """Carga configuración desde variables de entorno"""
        # RAG config
        if os.getenv('RAG_CHUNK_SIZE'):
            self.rag.chunk_size = int(os.getenv('RAG_CHUNK_SIZE'))
        if os.getenv('RAG_CHUNK_OVERLAP'):
            self.rag.chunk_overlap = int(os.getenv('RAG_CHUNK_OVERLAP'))
        if os.getenv('RAG_TOP_K'):
            self.rag.top_k = int(os.getenv('RAG_TOP_K'))
        if os.getenv('RAG_MAX_CONTEXT_TOKENS'):
            self.rag.max_context_tokens = int(os.getenv('RAG_MAX_CONTEXT_TOKENS'))
        
        # LLM config
        if os.getenv('LLM_MAX_TOKENS'):
            self.llm.max_tokens = int(os.getenv('LLM_MAX_TOKENS'))
        if os.getenv('LLM_TEMPERATURE'):
            self.llm.temperature = float(os.getenv('LLM_TEMPERATURE'))
        if os.getenv('LLM_TOP_P'):
            self.llm.top_p = float(os.getenv('LLM_TOP_P'))
        if os.getenv('LLM_REPEAT_PENALTY'):
            self.llm.repeat_penalty = float(os.getenv('LLM_REPEAT_PENALTY'))
    
    def save_to_file(self, config_file: str):
        """
        Guarda configuración actual a archivo JSON
        
        Args:
            config_file: Ruta donde guardar el archivo
        """
        try:
            data = {
                'rag': asdict(self.rag),
                'llm': asdict(self.llm)
            }
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Could not save config to {config_file}: {e}")
    
    def __repr__(self) -> str:
        """Representación string de la configuración"""
        return f"Config(rag={self.rag}, llm={self.llm})"


# Instancia global para uso conveniente (opcional)
default_config = Config()

