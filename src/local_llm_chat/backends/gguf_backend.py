"""
GGUF Backend - Backend para modelos GGUF vía llama-cpp-python

Implementa ModelBackend para modelos cuantizados GGUF usando llama.cpp.
Soporta GPU (CUDA/Metal) y aceleración automática.
"""
import os
import time
from typing import Dict, List, Optional, Any

from .base import ModelBackend


class GGUFBackend(ModelBackend):
    """
    Backend para modelos GGUF usando llama-cpp-python
    
    Características:
    - Carga modelos locales .gguf
    - Soporte GPU automático (CUDA/Metal)
    - Detección automática de formato de chat
    - Inferencia optimizada
    
    Uso:
        backend = GGUFBackend(
            model_path="models/llama-3.2-3b.gguf",
            n_gpu_layers=-1,  # Todas las capas en GPU
            n_ctx=8192
        )
        backend.load_model()
        
        response = backend.generate(
            messages=[{"role": "user", "content": "Hola"}],
            max_tokens=256
        )
    """
    
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        verbose: bool = False,
        **kwargs
    ):
        """
        Inicializa el backend GGUF
        
        Args:
            model_path: Ruta al archivo .gguf
            n_gpu_layers: Capas en GPU (-1 = todas, 0 = solo CPU)
            n_ctx: Tamaño del contexto
            verbose: Activar logs verbosos de llama.cpp
            **kwargs: Parámetros adicionales
        """
        super().__init__(**kwargs)
        
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        
        self.llm = None
        self._is_loaded = False
        
        # Se configurará en load_model()
        self.model_type = None
        self.supports_native_system = None
        self.chat_format = None
    
    def load_model(self) -> bool:
        """
        Carga el modelo GGUF en memoria
        
        Returns:
            True si exitoso, False si falla
        """
        try:
            from llama_cpp import Llama
            import torch
            from ..model_config import (
                detect_model_type,
                get_chat_format,
                supports_native_system,
            )
            
            print(f"[GGUF] Loading model: {os.path.basename(self.model_path)}")
            
            # Detectar tipo de modelo
            self.model_type = detect_model_type(self.model_path)
            self.chat_format = get_chat_format(self.model_type)
            self.supports_native_system = supports_native_system(self.model_type)
            
            print(f"[GGUF] Model type: {self.model_type}")
            print(f"[GGUF] Chat format: {self.chat_format or 'auto'}")
            print(f"[GGUF] Native system support: {'YES' if self.supports_native_system else 'NO (workaround)'}")
            
            # Configurar GPU automáticamente
            n_gpu = self.n_gpu_layers
            if n_gpu == -1:
                n_gpu = -1 if torch.cuda.is_available() else 0
            
            print(f"[GGUF] GPU layers: {n_gpu}")
            
            # Cargar modelo
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu,
                chat_format=self.chat_format,
                verbose=self.verbose,
            )
            
            self._is_loaded = True
            print("[GGUF] ✓ Model loaded successfully")
            return True
            
        except ImportError as e:
            print(f"[GGUF ERROR] llama-cpp-python not installed: {e}")
            print("[INFO] Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            print(f"[GGUF ERROR] Failed to load model: {e}")
            print("[INFO] Check if file is a valid GGUF model")
            return False
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera respuesta usando llama.cpp
        
        Args:
            messages: Historial de mensajes
            max_tokens: Máximo de tokens
            temperature: Temperatura
            top_p: Nucleus sampling
            repeat_penalty: Penalización de repetición
            top_k: Top-k sampling
            **kwargs: Parámetros adicionales
            
        Returns:
            dict con content, usage y elapsed_seconds
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Generar respuesta
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
            )
            
            elapsed = time.time() - start_time
            
            # Extraer contenido y métricas
            content = output["choices"][0]["message"]["content"]
            usage = output.get("usage", {})
            
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "elapsed_seconds": round(elapsed, 2)
            }
            
        except Exception as e:
            print(f"[GGUF ERROR] Generation failed: {e}")
            raise
    
    def unload_model(self):
        """Descarga el modelo de memoria"""
        if self.llm:
            del self.llm
            self.llm = None
            self._is_loaded = False
            print("[GGUF] Model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo"""
        return {
            "backend": "GGUF",
            "model_path": self.model_path,
            "model_name": os.path.basename(self.model_path),
            "model_type": self.model_type,
            "chat_format": self.chat_format,
            "supports_native_system": self.supports_native_system,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "is_loaded": self._is_loaded,
        }
    
    @property
    def is_loaded(self) -> bool:
        """Indica si el modelo está cargado"""
        return self._is_loaded

