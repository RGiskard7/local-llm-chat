"""
Cliente de Chat Universal - Interfaz multi-modelo LLM
Clase principal para gestionar sesiones de chat con modelos GGUF
"""

import os
import json
from datetime import datetime
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import torch

from .model_config import (
    detect_model_type,
    get_chat_format,
    supports_native_system,
    get_model_info,
)

# Configuración
LOGS_DIR = "./chat_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

class UniversalChatClient:
    """
    Cliente de chat universal que soporta cualquier modelo GGUF con adaptación automática de system prompt.

    Args:
        model_key: Clave de modelo popular (ej., "gemma-12b", "llama-3-8b")
        model_path: Ruta directa al modelo GGUF local
        repo_id: Repositorio de Hugging Face (si no se usa model_key)
        filename: Nombre de archivo GGUF (si no se usa model_key)
        n_gpu_layers: Número de capas en GPU (-1 = todas)
        system_prompt: Prompt del sistema inicial (opcional)
    """

    def __init__(
        self, 
        model_key=None, 
        model_path=None, 
        repo_id=None, 
        filename=None,
        n_gpu_layers=-1, 
        system_prompt=None
    ):
        """Inicializa el cliente de chat universal."""

        print("[INIT] Universal Chat Client v1.0")
        print("=" * 60)

        # Determinar qué modelo cargar
        if model_key:
            model_info = get_model_info(model_key)
            if not model_info:
                raise ValueError(f"Model '{model_key}' not found. Use list_models() for options.")

            print(f"[MODEL] {model_info['description']}")
            repo_id = model_info["repo_id"]
            filename = model_info["filename"]
            self.model_type = model_info["type"]

        elif model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model_type = detect_model_type(model_path)

        elif repo_id and filename:
            self.model_type = detect_model_type(filename)

        else:
            raise ValueError(
                "No model specified. Please specify a model using one of:\n"
                "  - model_key (e.g., 'gemma-12b', 'llama-3-8b')\n"
                "  - model_path (path to local .gguf file)\n"
                "  - repo_id and filename (Hugging Face repository)\n"
                "Use /models command to see available models and recommendations."
            )

        # Descargar modelo si es necesario
        if not model_path:
            print(f"[DOWNLOAD] {filename}")
            print("[INFO] First download may take several minutes...")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir="./models",
            )

        # Detectar capacidades del modelo
        self.supports_native_system = supports_native_system(self.model_type)
        self.chat_format = get_chat_format(self.model_type)

        print(f"[DETECT] Model type: {self.model_type}")
        print(f"[DETECT] Chat format: {self.chat_format or 'auto'}")
        print(f"[DETECT] Native system support: {'YES' if self.supports_native_system else 'NO (using workaround)'}")

        # Configurar GPU
        if n_gpu_layers == -1:
            n_gpu_layers = -1 if torch.cuda.is_available() else 0

        print(f"[GPU] Layers: {n_gpu_layers}")
        print("[LOAD] Loading model into memory...")

        # Cargar modelo con manejo de errores
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=8192,
                n_gpu_layers=n_gpu_layers,
                chat_format=self.chat_format,
                verbose=False,
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to load model: {e}")
            print(f"[INFO] Common issues:")
            print(f"  - File is corrupted (try re-downloading)")
            print(f"  - Insufficient RAM/VRAM (try a smaller model)")
            print(f"  - Incompatible GGUF version (update llama-cpp-python)")
            raise

        # Configuración del prompt del sistema
        self.system_prompt = system_prompt
        self.preset_name = None

        # Historial de conversación
        self.conversation_history = []
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

        print("[READY] Model loaded successfully")
        print(f"[SESSION] ID: {self.session_id}")

        if self.system_prompt:
            preview = self.system_prompt[:80] + "..." if len(self.system_prompt) > 80 else self.system_prompt
            print(f"[SYSTEM] Active prompt: {preview}")

        print("=" * 60)

    def set_system_prompt(self, prompt: str):
        """Establece el prompt del sistema."""
        self.system_prompt = prompt
        self.preset_name = None
        preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        print(f"[SYSTEM] Prompt updated: {preview}")

    def clear_system_prompt(self):
        """Elimina el prompt del sistema."""
        self.system_prompt = None
        self.preset_name = None
        print("[SYSTEM] Prompt cleared")

    def show_system_prompt(self):
        """Muestra el prompt del sistema actual."""
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
        """Lista los presets disponibles."""
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
        Construye la lista de mensajes con adaptación del prompt del sistema.

        Args:
            prompt: Mensaje actual del usuario

        Returns:
            Lista de mensajes formateada para el modelo
        """
        messages = []

        if self.system_prompt:
            if self.supports_native_system:
                # Modelos con soporte nativo de system (Llama, Mistral, etc.)
                messages.append({"role": "system", "content": self.system_prompt})
            else:
                # Modelos sin soporte nativo (Gemma, Phi, etc.)
                messages.append({"role": "user", "content": self.system_prompt})
                messages.append({"role": "assistant", "content": "Understood. I will follow these instructions."})

        # Agregar historial de conversación
        for entry in self.conversation_history:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})

        # Agregar mensaje actual
        messages.append({"role": "user", "content": prompt})

        return messages

    def infer(self, prompt: str, max_tokens: int = 256) -> str:
        """Genera la respuesta del modelo."""
        import time

        messages = self._build_messages_with_system(prompt)

        start_time = time.time()
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        elapsed = time.time() - start_time

        response = out["choices"][0]["message"]["content"]

        # Extraer uso de tokens si está disponible
        usage = out.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Mostrar métricas
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        print(f"[METRICS] {time_str} | IN:{prompt_tokens} OUT:{completion_tokens} TOTAL:{total_tokens}")

        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": prompt,
            "assistant": response,
            "metrics": {
                "elapsed_seconds": round(elapsed, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        })

        return response

    def save_log(self):
        """Guarda la conversación en un archivo JSON."""
        filename_parts = [self.session_id]

        filename_parts.append(self.model_type)

        if self.preset_name:
            filename_parts.append(self.preset_name)
        elif self.conversation_history:
            first_message = self.conversation_history[0]["user"]
            words = first_message.split()[:4]
            preview = "_".join(words)
            preview = "".join(c if c.isalnum() or c in "_ " else "" for c in preview)
            preview = preview.replace(" ", "_")[:30]
            if preview:
                filename_parts.append(preview)

        filename = "_".join(filename_parts) + ".json"
        log_file = os.path.join(LOGS_DIR, filename)

        log_data = {
            "session_id": self.session_id,
            "model_type": self.model_type,
            "chat_format": self.chat_format,
            "supports_native_system": self.supports_native_system,
            "preset_name": self.preset_name,
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "total_messages": len(self.conversation_history),
            "conversation": self.conversation_history
        }

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"\n[SAVE] Log saved: {log_file}")

    def show_history(self):
        """Muestra el historial de conversación."""
        if not self.conversation_history:
            print("\n[INFO] No history available")
            return

        print("\n" + "=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)
        for i, entry in enumerate(self.conversation_history, 1):
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
        """Muestra las estadísticas de la sesión."""
        if not self.conversation_history:
            print("\n[INFO] No statistics available")
            return

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

        avg_time = total_time / len(self.conversation_history) if self.conversation_history else 0

        print("\n" + "=" * 60)
        print("SESSION STATISTICS")
        print("=" * 60)
        print(f"Messages: {len(self.conversation_history)}")
        print(f"Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
        print(f"Avg time/message: {avg_time:.1f}s")
        print(f"Total tokens: {total_tokens}")
        print(f"  Input: {total_prompt_tokens}")
        print(f"  Output: {total_completion_tokens}")
        print("=" * 60 + "\n")

    def change_model(self, model_path: str):
        """Cambia a un modelo diferente."""
        # Validar que el archivo existe
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            return False

        # Validar que el archivo es GGUF
        if not model_path.endswith('.gguf'):
            print(f"[ERROR] Invalid file type. Expected .gguf file")
            return False

        print(f"\n[LOAD] Switching to: {os.path.basename(model_path)}")

        # Guardar sesión actual si hay historial
        if self.conversation_history:
            self.save_log()
            print("[INFO] Previous session saved")

        # Detectar nuevo modelo
        self.model_type = detect_model_type(model_path)
        self.supports_native_system = supports_native_system(self.model_type)
        self.chat_format = get_chat_format(self.model_type)

        print(f"[DETECT] Model type: {self.model_type}")
        print(f"[DETECT] Native system support: {'YES' if self.supports_native_system else 'NO (using workaround)'}")

        # Advertir si el tipo de modelo es desconocido
        if self.model_type == "unknown":
            print("[WARN] Model type not recognized. Using generic settings.")
            print("[WARN] Response quality may vary. Consider adding to model_config.py")

        # Descargar modelo antiguo
        del self.llm

        # Cargar nuevo modelo con manejo de errores
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=8192,
                n_gpu_layers=-1 if torch.cuda.is_available() else 0,
                chat_format=self.chat_format,
                verbose=False,
            )
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print(f"[INFO] Check if file is a valid GGUF model")
            print(f"[INFO] You may need to re-download or try a different model")
            return False

        # Reiniciar sesión (nuevo modelo = nueva conversación)
        self.conversation_history = []
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

        print(f"[READY] Model loaded successfully")
        print(f"[SESSION] New ID: {self.session_id}")

        # Mantener el prompt del sistema si está configurado
        if self.system_prompt:
            print(f"[SYSTEM] Keeping active prompt")

        print("=" * 60 + "\n")
        return True
