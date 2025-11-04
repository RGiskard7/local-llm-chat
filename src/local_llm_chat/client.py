"""
Cliente de Chat Universal - Interfaz multi-backend LLM
Clase principal para gestionar sesiones de chat con diferentes backends de modelos
Soporta: GGUF (llama-cpp-python) y Transformers (Hugging Face)
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from huggingface_hub import hf_hub_download

from .backends import GGUFBackend, TransformersBackend, TRANSFORMERS_AVAILABLE
from .model_config import get_model_info, detect_backend_type

# Configuración
LOGS_DIR = "./chat_logs"
os.makedirs(LOGS_DIR, exist_ok=True)


class ConversationManager:
    """
    Gestiona el historial de conversación y métricas de sesión.
    Responsabilidad única: tracking de conversaciones.
    """
    
    def __init__(self, session_id: str = None):
        self.conversation_history: List[Dict] = []
        self.session_start = datetime.now()
        self.session_id = session_id or self.session_start.strftime("%Y%m%d_%H%M%S")
    
    def add_message(self, user_msg: str, assistant_msg: str, metrics: Dict[str, Any]):
        """Añade un intercambio al historial"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "assistant": assistant_msg,
            "metrics": metrics
        })
    
    def clear_history(self):
        """Limpia el historial"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Retorna el historial completo"""
        return self.conversation_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Calcula estadísticas de la sesión"""
        if not self.conversation_history:
            return {}
        
        total_time = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        for entry in self.conversation_history:
            if 'metrics' in entry:
                m = entry['metrics']
                total_time += m.get('elapsed_seconds', 0)
                total_prompt_tokens += m.get('prompt_tokens', 0)
                total_completion_tokens += m.get('completion_tokens', 0)
                total_tokens += m.get('total_tokens', 0)
        
        avg_time = total_time / len(self.conversation_history)
        
        return {
            "messages": len(self.conversation_history),
            "total_time": total_time,
            "avg_time": avg_time,
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }
    
    def reset_session(self, session_id: str = None):
        """Reinicia la sesión con un nuevo ID"""
        self.conversation_history = []
        self.session_start = datetime.now()
        self.session_id = session_id or self.session_start.strftime("%Y%m%d_%H%M%S")


class UniversalChatClient:
    """
    Cliente de chat universal que soporta múltiples backends de modelos con interfaz común
    
    Soporta:
    - GGUF: Modelos cuantizados vía llama-cpp-python
    - Transformers: Modelos Hugging Face (local o remoto)
    
    Características:
    - Detección automática de tipo de modelo
    - System prompts adaptativos según capacidades
    - Presets configurables
    - Gestión de historial y sesiones
    - Cambio dinámico de modelos y backends
    
    Uso GGUF:
        client = UniversalChatClient(
            backend="gguf",
            model_path="models/llama-3.2-3b.gguf"
        )
    
    Uso Transformers:
        client = UniversalChatClient(
            backend="transformers",
            model_name_or_path="bigscience/bloom-560m"
        )
    
    Uso con model_key (legacy):
        client = UniversalChatClient(
            model_key="llama-3-8b"
        )
    """
    
    def __init__(
        self,
        # Parámetros de backend
        backend: str = "gguf",  # "gguf" o "transformers"
        
        # Parámetros GGUF
        model_key: Optional[str] = None,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        
        # Parámetros Transformers
        model_name_or_path: Optional[str] = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        
        # Parámetros comunes
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        llm_config=None,
    ):
        """
        Inicializa el cliente de chat universal
        
        Args:
            backend: Tipo de backend ("gguf" o "transformers")
            
            # GGUF params
            model_key: Clave de modelo popular (legacy)
            model_path: Ruta al modelo (funciona con ambos backends)
            repo_id: Repositorio HF (para GGUF)
            filename: Nombre archivo GGUF
            n_gpu_layers: Capas en GPU para GGUF
            n_ctx: Tamaño contexto para GGUF
            
            # Transformers params
            model_name_or_path: Alias de model_path (convención HuggingFace)
            device: Dispositivo ("auto", "cuda", "cpu")
            torch_dtype: Tipo de datos ("auto", "float16", etc.)
            trust_remote_code: Permitir código remoto
            load_in_8bit: Cargar en 8-bit
            load_in_4bit: Cargar en 4-bit
            
            # Common params
            system_prompt: System prompt inicial
            verbose: Logs verbosos
            llm_config: Configuración LLM (opcional, para infer())
        
        Nota sobre model_path vs model_name_or_path:
            Ambos parámetros son INTERCAMBIABLES y funcionan con cualquier backend.
            - model_path: Nombre más genérico
            - model_name_or_path: Convención HuggingFace (más descriptivo)
            
            Usa el que prefieras según tu caso:
            - GGUF: model_path="models/llama.gguf" (recomendado)
            - Transformers: model_name_or_path="bigscience/bloom" (recomendado)
            
            Pero ambos funcionan con ambos backends.
        """
        print("[INIT] Universal Chat Client v2.0 - Multi-Backend")
        
        # Guardar config LLM para infer() (sin cargarla aquí)
        self.llm_config = llm_config
        
        # Guardar parámetros del modelo para change_model()
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        print("=" * 60)
        
        self.backend_type = backend.lower()
        self.verbose = verbose
        self.backend = None
        
        # ALIAS RESOLUTION: Unificar model_path y model_name_or_path
        # Ambos parámetros hacen lo mismo, solo difieren en el nombre
        if model_path and model_name_or_path:
            raise ValueError(
                "Cannot specify both 'model_path' and 'model_name_or_path'. "
                "They are aliases - use only one.\n"
                f"  model_path: {model_path}\n"
                f"  model_name_or_path: {model_name_or_path}"
            )
        
        # Si se usó model_name_or_path, copiarlo a model_path
        if model_name_or_path and not model_path:
            model_path = model_name_or_path
        
        # A partir de aquí, solo usamos model_path internamente
        
        # Configuración del prompt del sistema
        self.system_prompt = system_prompt
        self.preset_name = None
        
        # Gestión de conversación (delegada a ConversationManager)
        self._conversation = ConversationManager()
        
        # Propiedades públicas para compatibilidad de API
        self.conversation_history = self._conversation.conversation_history
        self.session_start = self._conversation.session_start
        self.session_id = self._conversation.session_id
        
        # Inicializar backend según tipo
        if self.backend_type == "gguf":
            self._init_gguf_backend(
                model_key, model_path, repo_id, filename,
                n_gpu_layers, n_ctx
            )
        elif self.backend_type == "transformers":
            self._init_transformers_backend(
                model_name_or_path, device, torch_dtype,
                trust_remote_code, load_in_8bit, load_in_4bit
            )
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Supported: 'gguf', 'transformers'"
            )
        
        # Información del modelo desde backend
        if self.backend and self.backend.is_loaded:
            model_info = self.backend.get_model_info()
            self.model_type = model_info.get("model_type", "unknown")
            self.supports_native_system = model_info.get("supports_native_system", False)
            self.chat_format = model_info.get("chat_format")
            
            print("[READY] Model loaded successfully")
            print(f"[SESSION] ID: {self._conversation.session_id}")
            
            if self.system_prompt:
                preview = self.system_prompt[:80] + "..." if len(self.system_prompt) > 80 else self.system_prompt
                print(f"[SYSTEM] Active prompt: {preview}")
        
        print("=" * 60)
    
    def _init_gguf_backend(
        self,
        model_key: Optional[str],
        model_path: Optional[str],
        repo_id: Optional[str],
        filename: Optional[str],
        n_gpu_layers: int,
        n_ctx: int
    ):
        """Inicializa backend GGUF"""
        print("[BACKEND] Using GGUF (llama-cpp-python)")
        
        # Determinar qué modelo cargar (legacy model_key support)
        if model_key:
            model_info = get_model_info(model_key)
            if not model_info:
                raise ValueError(f"Model '{model_key}' not found. Use list_models() for options.")
            
            print(f"[MODEL] {model_info['description']}")
            repo_id = model_info["repo_id"]
            filename = model_info["filename"]
        
        # Descargar modelo si es necesario
        if not model_path:
            if not (repo_id and filename):
                raise ValueError(
                    "For GGUF backend, provide either:\n"
                    "  - model_path (local .gguf file)\n"
                    "  - model_key (popular model)\n"
                    "  - repo_id + filename (HuggingFace)"
                )
            
            print(f"[DOWNLOAD] {filename}")
            print("[INFO] First download may take several minutes...")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir="./models",
            )
        
        # Validar que el archivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Crear backend y cargar modelo
        self.backend = GGUFBackend(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=self.verbose
        )
        
        success = self.backend.load_model()
        if not success:
            raise RuntimeError("Failed to load GGUF model")
        
        # Obtener información del modelo del backend
        model_info = self.backend.get_model_info()
        self.model_type = model_info.get("model_type", "unknown")
        self.supports_native_system = model_info.get("supports_native_system", False)
        self.chat_format = model_info.get("chat_format")
    
    def _init_transformers_backend(
        self,
        model_path: Optional[str],
        device: str,
        torch_dtype: str,
        trust_remote_code: bool,
        load_in_8bit: bool,
        load_in_4bit: bool
    ):
        """Inicializa backend Transformers"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "Transformers backend not available. "
                "Install with: pip install transformers accelerate"
            )
        
        print("[BACKEND] Using Transformers (Hugging Face)")
        
        if not model_path:
            raise ValueError(
                "For Transformers backend, provide:\n"
                "  - model_path or model_name_or_path (HuggingFace name or local path)\n"
                "Examples: 'bigscience/bloom-560m', 'meta-llama/Llama-2-7b-hf'"
            )
        
        # Crear backend y cargar modelo
        self.backend = TransformersBackend(
            model_name_or_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        
        success = self.backend.load_model()
        if not success:
            raise RuntimeError("Failed to load Transformers model")
    
    def set_system_prompt(self, prompt: str):
        """Establece el prompt del sistema"""
        self.system_prompt = prompt
        self.preset_name = None
        preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        print(f"[SYSTEM] Prompt updated: {preview}")
    
    def clear_system_prompt(self):
        """Elimina el prompt del sistema"""
        self.system_prompt = None
        self.preset_name = None
        print("[SYSTEM] Prompt cleared")
    
    def show_system_prompt(self):
        """Muestra el prompt del sistema actual"""
        if self.system_prompt:
            print("\n" + "=" * 60)
            print("ACTIVE SYSTEM PROMPT")
            print("=" * 60)
            print(self.system_prompt)
            print("=" * 60 + "\n")
        else:
            print("[WARN] No system prompt configured")
    
    def load_preset(self, preset_name: str) -> bool:
        """Carga un preset desde prompts.py"""
        try:
            from .prompts import PROMPTS
        except ImportError:
            print("[ERROR] prompts.py not available")
            return False
        
        if preset_name not in PROMPTS:
            print(f"[ERROR] Preset '{preset_name}' not found")
            print(f"[INFO] Available: {', '.join(PROMPTS.keys())}")
            return False
        
        self.system_prompt = PROMPTS[preset_name]
        self.preset_name = preset_name
        print(f"[SYSTEM] Preset '{preset_name}' loaded")
        return True
    
    def list_presets(self):
        """Lista los presets disponibles"""
        try:
            from .prompts import PROMPTS
        except ImportError:
            print("[WARN] prompts.py not available")
            return
        
        print("\n" + "=" * 60)
        print("AVAILABLE PRESETS")
        print("=" * 60)
        for i, name in enumerate(PROMPTS.keys(), 1):
            print(f"  {i}. {name}")
        print("=" * 60 + "\n")
    
    def _build_messages_with_system(self, prompt: str) -> list:
        """
        Construye la lista de mensajes con adaptación del prompt del sistema
        
        Args:
            prompt: Mensaje actual del usuario
            
        Returns:
            Lista de mensajes formateada según el backend
        """
        messages = []
        
        # Agregar historial de conversación
        for entry in self._conversation.conversation_history:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        
        # Agregar mensaje actual
        messages.append({"role": "user", "content": prompt})
        
        # Formatear con system prompt usando el backend
        if self.system_prompt:
            messages = self.backend.format_messages(messages, self.system_prompt)
        
        return messages
    
    def infer(
        self, 
        prompt: str, 
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        repeat_penalty: float = None,
        top_k: int = None,
        **kwargs
    ) -> str:
        """
        Genera respuesta del modelo
        
        Args:
            prompt: Mensaje del usuario
            max_tokens: Máximo de tokens a generar (> 0)
            temperature: Temperatura para sampling (>= 0)
            top_p: Nucleus sampling (0-1)
            repeat_penalty: Penalización de repetición (> 0)
            top_k: Top-k sampling (> 0)
            **kwargs: Parámetros adicionales específicos del backend
            
        Returns:
            Respuesta del modelo
            
        Raises:
            RuntimeError: Si el modelo no está cargado
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones básicas
        if not self.backend or not self.backend.is_loaded:
            raise RuntimeError("Model not loaded")
        
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")
        
        # Usar valores de llm_config si está disponible y no se especifican
        if self.llm_config:
            if max_tokens is None:
                max_tokens = self.llm_config.max_tokens
            if temperature is None:
                temperature = self.llm_config.temperature
            if top_p is None:
                top_p = self.llm_config.top_p
            if repeat_penalty is None:
                repeat_penalty = self.llm_config.repeat_penalty
            if top_k is None:
                top_k = getattr(self.llm_config, 'top_k', 40)
        else:
            # Fallback a valores por defecto si no hay config
            if max_tokens is None:
                max_tokens = 256
            if temperature is None:
                temperature = 0.7
            if top_p is None:
                top_p = 0.9
            if repeat_penalty is None:
                repeat_penalty = 1.1
            if top_k is None:
                top_k = 40
        
        # Validar rangos de parámetros
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not (0 <= top_p <= 1):
            raise ValueError(f"top_p must be between 0 and 1, got {top_p}")
        if repeat_penalty <= 0:
            raise ValueError(f"repeat_penalty must be > 0, got {repeat_penalty}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        
        # Construir mensajes
        messages = self._build_messages_with_system(prompt)
        
        # Generar respuesta usando el backend
        result = self.backend.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            **kwargs
        )
        
        response = result["content"]
        usage = result["usage"]
        elapsed = result["elapsed_seconds"]
        
        # Mostrar métricas
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        
        print(f"[METRICS] {time_str} | IN:{usage['prompt_tokens']} OUT:{usage['completion_tokens']} TOTAL:{usage['total_tokens']}")
        
        # Guardar en historial usando ConversationManager
        self._conversation.add_message(
            user_msg=prompt,
            assistant_msg=response,
            metrics={
                "elapsed_seconds": elapsed,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"]
            }
        )
        
        return response
    
    def save_log(self):
        """Guarda la conversación en un archivo JSON"""
        history = self._conversation.get_history()
        filename_parts = [self._conversation.session_id]
        
        filename_parts.append(self.model_type)
        
        if self.preset_name:
            filename_parts.append(self.preset_name)
        elif history:
            first_message = history[0]["user"]
            words = first_message.split()[:4]
            preview = "_".join(words)
            preview = "".join(c if c.isalnum() or c in "_ " else "" for c in preview)
            preview = preview.replace(" ", "_")[:30]
            if preview:
                filename_parts.append(preview)
        
        filename = "_".join(filename_parts) + ".json"
        log_file = os.path.join(LOGS_DIR, filename)
        
        model_info = self.backend.get_model_info() if self.backend else {}
        
        log_data = {
            "session_id": self._conversation.session_id,
            "backend_type": self.backend_type,
            "model_type": self.model_type,
            "model_info": model_info,
            "chat_format": self.chat_format,
            "supports_native_system": self.supports_native_system,
            "preset_name": self.preset_name,
            "session_start": self._conversation.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "total_messages": len(history),
            "conversation": history
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] Log saved: {log_file}")
    
    def show_history(self):
        """Muestra el historial de conversación"""
        history = self._conversation.get_history()
        if not history:
            print("\n[INFO] No history available")
            return
        
        print("\n" + "=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)
        for i, entry in enumerate(history, 1):
            print(f"\n[{i}] USER:")
            print(f"    {entry['user']}")
            print(f"\n    ASSISTANT:")
            print(f"    {entry['assistant']}")
            
            # Mostrar métricas si están disponibles
            if 'metrics' in entry:
                m = entry['metrics']
                elapsed = m.get('elapsed_seconds', 0)
                if elapsed < 60:
                    time_str = f"{elapsed:.1f}s"
                else:
                    time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                print(f"    [{time_str} | IN:{m.get('prompt_tokens', 0)} OUT:{m.get('completion_tokens', 0)}]")
        
        print("\n" + "=" * 60)
    
    def show_stats(self):
        """Muestra las estadísticas de la sesión"""
        stats = self._conversation.get_stats()
        if not stats:
            print("\n[INFO] No statistics available")
            return
        
        total_time = stats['total_time']
        avg_time = stats['avg_time']
        total_tokens = stats['total_tokens']
        total_prompt_tokens = stats['prompt_tokens']
        total_completion_tokens = stats['completion_tokens']
        
        print("\n" + "=" * 60)
        print("SESSION STATISTICS")
        print("=" * 60)
        print(f"Backend: {self.backend_type}")
        print(f"Model: {self.model_type}")
        print(f"Messages: {stats['messages']}")
        print(f"Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
        print(f"Avg time/message: {avg_time:.1f}s")
        print(f"Total tokens: {total_tokens}")
        print(f"  Input: {total_prompt_tokens}")
        print(f"  Output: {total_completion_tokens}")
        print("=" * 60 + "\n")
    
    def change_model(
        self,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Cambia a un modelo diferente (puede cambiar de backend)
        
        Args:
            model_path: Ruta al nuevo modelo (obligatorio)
            backend: Nuevo backend ("gguf" o "transformers", None = auto-detectar)
            **kwargs: Parámetros específicos del backend
            
        Returns:
            True si exitoso, False si falla
            
        Ejemplos:
            # Cambiar a otro GGUF
            client.change_model(model_path="models/mistral-7b.gguf")
            
            # Cambiar a Transformers
            client.change_model(
                backend="transformers",
                model_name_or_path="bigscience/bloom-560m"
            )
        """
        # Guardar sesión actual si hay historial
        if self._conversation.get_history():
            self.save_log()
            print("[INFO] Previous session saved")
        
        # Obtener model_path de kwargs si no se pasó explícitamente (compatibilidad)
        if not model_path:
            model_path = kwargs.get("model_path") or kwargs.get("model_name_or_path")
        
        if not model_path:
            raise ValueError("model_path (or model_name_or_path) is required")
        
        # Si se pasa solo model_path sin backend, detectar automáticamente
        if backend is None:
            backend = detect_backend_type(model_path)
        
        # Descargar modelo anterior de memoria (fix memory leak)
        if self.backend:
            try:
                print("[INFO] Unloading previous model...")
                self.backend.unload_model()
            except Exception as e:
                print(f"[WARN] Failed to unload previous model: {e}")
        
        # Determinar backend
        new_backend = backend if backend else self.backend_type
        
        print(f"\n[CHANGE] Switching to backend: {new_backend}")
        
        try:
            # Reiniciar configuración
            self.backend_type = new_backend
            self._conversation.reset_session()
            # Actualizar referencias públicas
            self.conversation_history = self._conversation.conversation_history
            self.session_start = self._conversation.session_start
            self.session_id = self._conversation.session_id
            
            # Inicializar nuevo backend
            if new_backend == "gguf":
                if not model_path:
                    raise ValueError("model_path required for GGUF backend")
                
                self._init_gguf_backend(
                    model_key=None,
                    model_path=model_path,
                    repo_id=kwargs.get("repo_id"),
                    filename=kwargs.get("filename"),
                    n_gpu_layers=kwargs.get("n_gpu_layers", self.n_gpu_layers),
                    n_ctx=kwargs.get("n_ctx", self.n_ctx)
                )
            elif new_backend == "transformers":
                # model_path ya está definido arriba, solo validar
                if not model_path:
                    raise ValueError(
                        "model_path or model_name_or_path required for Transformers backend"
                    )
                
                self._init_transformers_backend(
                    model_path=model_path,
                    device=kwargs.get("device", "auto"),
                    torch_dtype=kwargs.get("torch_dtype", "auto"),
                    trust_remote_code=kwargs.get("trust_remote_code", False),
                    load_in_8bit=kwargs.get("load_in_8bit", False),
                    load_in_4bit=kwargs.get("load_in_4bit", False)
                )
            else:
                raise ValueError(f"Unknown backend: {new_backend}")
            
            # Actualizar información del modelo
            if self.backend and self.backend.is_loaded:
                model_info = self.backend.get_model_info()
                self.model_type = model_info.get("model_type", "unknown")
                self.supports_native_system = model_info.get("supports_native_system", False)
                self.chat_format = model_info.get("chat_format")
            
            print(f"[READY] Model loaded successfully")
            print(f"[SESSION] New ID: {self._conversation.session_id}")
            
            # Mantener el prompt del sistema si está configurado
            if self.system_prompt:
                print(f"[SYSTEM] Keeping active prompt")
            
            print("=" * 60 + "\n")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to change model: {e}")
            import traceback
            traceback.print_exc()
            return False
