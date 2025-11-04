"""
Transformers Backend - Backend para modelos Hugging Face Transformers

Implementa ModelBackend para modelos de Hugging Face (BERT, GPT, Llama, etc.).
Soporta modelos locales y remotos, con aceleración GPU automática.
"""
import os
import time
from typing import Dict, List, Optional, Any

from .base import ModelBackend


class TransformersBackend(ModelBackend):
    """
    Backend para modelos Hugging Face Transformers
    
    Características:
    - Carga modelos desde HuggingFace Hub o locales
    - Soporte multi-arquitectura (BERT, GPT, Llama, Mistral, etc.)
    - Aceleración GPU automática
    - Detección de capacidades de chat
    
    Uso:
        # Modelo remoto
        backend = TransformersBackend(
            model_name_or_path="bigscience/bloom-560m",
            device="auto"
        )
        
        # Modelo local
        backend = TransformersBackend(
            model_name_or_path="/path/to/local/model",
            device="cuda"
        )
        
        backend.load_model()
        response = backend.generate(
            messages=[{"role": "user", "content": "Hola"}],
            max_tokens=256
        )
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Inicializa el backend Transformers
        
        Args:
            model_name_or_path: Nombre HF o path local
            device: Dispositivo ("auto", "cuda", "cpu")
            torch_dtype: Tipo de datos ("auto", "float16", "bfloat16", "float32")
            trust_remote_code: Permitir código remoto personalizado
            load_in_8bit: Cargar en 8-bit (requiere bitsandbytes)
            load_in_4bit: Cargar en 4-bit (requiere bitsandbytes)
            **kwargs: Parámetros adicionales
        """
        super().__init__(**kwargs)
        
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # Detectar tipo de modelo (se actualiza en load_model)
        self.model_type = self._detect_model_type(model_name_or_path)
        self.supports_native_system = self._supports_system_prompt()
        self.chat_format = "transformers"
    
    def _detect_model_type(self, model_name: str) -> str:
        """
        Detecta el tipo de modelo basándose en el nombre
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Tipo detectado (llama, mistral, gpt, bert, etc.)
        """
        name_lower = model_name.lower()
        
        # Familias conocidas
        if "llama" in name_lower:
            if "llama-3" in name_lower or "llama3" in name_lower:
                return "llama-3"
            elif "llama-2" in name_lower or "llama2" in name_lower:
                return "llama-2"
            return "llama"
        elif "mistral" in name_lower or "mixtral" in name_lower:
            return "mistral"
        elif "gemma" in name_lower:
            return "gemma"
        elif "phi" in name_lower:
            return "phi"
        elif "gpt" in name_lower:
            return "gpt"
        elif "bloom" in name_lower:
            return "bloom"
        elif "falcon" in name_lower:
            return "falcon"
        elif "bert" in name_lower:
            return "bert"
        elif "qwen" in name_lower:
            return "qwen"
        
        return "unknown"
    
    def _supports_system_prompt(self) -> bool:
        """
        Determina si el modelo soporta system prompts nativamente
        
        Returns:
            True si soporta, False si requiere workaround
        """
        # Modelos que soportan nativamente system prompts
        native_support = ["llama", "mistral", "gpt", "qwen"]
        
        for model in native_support:
            if model in self.model_type.lower():
                return True
        
        return False
    
    def load_model(self) -> bool:
        """
        Carga el modelo Transformers en memoria
        
        Returns:
            True si exitoso, False si falla
        """
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer,
                BitsAndBytesConfig
            )
            import torch
            
            print(f"[TRANSFORMERS] Loading model: {self.model_name_or_path}")
            print(f"[TRANSFORMERS] Model type: {self.model_type}")
            print(f"[TRANSFORMERS] Native system support: {'YES' if self.supports_native_system else 'NO (workaround)'}")
            
            # Configurar cuantización si se solicita
            quantization_config = None
            if self.load_in_8bit or self.load_in_4bit:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=self.load_in_8bit,
                        load_in_4bit=self.load_in_4bit,
                    )
                    print(f"[TRANSFORMERS] Quantization: {'8-bit' if self.load_in_8bit else '4-bit'}")
                except ImportError:
                    print("[TRANSFORMERS WARNING] bitsandbytes not available, loading in full precision")
            
            # Configurar dtype
            dtype = self.torch_dtype
            if dtype == "auto":
                # Usar float16 si hay GPU (CUDA o MPS), sino float32
                has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
                dtype = torch.float16 if has_gpu else torch.float32
            elif dtype == "float16":
                dtype = torch.float16
            elif dtype == "bfloat16":
                dtype = torch.bfloat16
            elif dtype == "float32":
                dtype = torch.float32
            
            print(f"[TRANSFORMERS] Dtype: {dtype}")
            
            # Verificar si accelerate está disponible (requerido para device_map)
            try:
                import accelerate
                accelerate_available = True
            except ImportError:
                accelerate_available = False
            
            # Configurar device_map para HuggingFace
            # Nota: device_map="auto" requiere accelerate
            device_map = None
            if self.device == "auto":
                if accelerate_available:
                    # Usar device_map="auto" de HF para balanceo inteligente
                    if torch.cuda.is_available():
                        print(f"[TRANSFORMERS] Detected CUDA GPU - using HF auto device_map")
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        print(f"[TRANSFORMERS] Detected Metal (MPS) - using HF auto device_map")
                    else:
                        print(f"[TRANSFORMERS] No GPU detected - using CPU")
                    device_map = "auto"
                else:
                    # Fallback sin accelerate: cargar en dispositivo específico
                    print(f"[TRANSFORMERS] accelerate not installed, using fallback device selection")
                    if torch.cuda.is_available():
                        device_map = "cuda"
                        print(f"[TRANSFORMERS] Using CUDA")
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device_map = "mps"
                        print(f"[TRANSFORMERS] Using Metal (MPS)")
                    else:
                        device_map = "cpu"
                        print(f"[TRANSFORMERS] Using CPU")
            else:
                # Usuario especificó dispositivo explícito
                device_map = self.device
                print(f"[TRANSFORMERS] Using device: {device_map}")
            
            # Cargar tokenizer
            print("[TRANSFORMERS] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code
            )
            
            # Asegurar pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Cargar modelo
            print("[TRANSFORMERS] Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=self.trust_remote_code,
                quantization_config=quantization_config,
            )
            
            self.model.eval()  # Modo evaluación
            
            self._is_loaded = True
            print("[TRANSFORMERS] ✓ Model loaded successfully")
            
            # Información del dispositivo
            if hasattr(self.model, 'hf_device_map'):
                print(f"[TRANSFORMERS] Device map: {self.model.hf_device_map}")
            
            return True
            
        except ImportError as e:
            print(f"[TRANSFORMERS ERROR] transformers not installed: {e}")
            print("[INFO] Install with: pip install transformers accelerate")
            return False
        except Exception as e:
            print(f"[TRANSFORMERS ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repeat_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera respuesta usando Transformers
        
        Args:
            messages: Historial de mensajes
            max_tokens: Máximo de tokens nuevos
            temperature: Temperatura
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repeat_penalty: Penalización por repetición (HF: repetition_penalty)
            do_sample: Activar sampling (False = greedy)
            **kwargs: Parámetros adicionales
            
        Returns:
            dict con content, usage y elapsed_seconds
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        
        start_time = time.time()
        
        try:
            # Convertir mensajes a texto
            # Intentar usar chat template si está disponible
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    # Fallback a concatenación simple
                    prompt = self._messages_to_text(messages)
            else:
                prompt = self._messages_to_text(messages)
            
            # Tokenizar
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            # Mover a dispositivo del modelo
            if self.model.device.type != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            input_length = inputs['input_ids'].shape[1]
            
            # Generar
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    top_k=top_k if do_sample else 50,
                    repetition_penalty=repeat_penalty,  # HF usa repetition_penalty
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decodificar solo los tokens nuevos
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            elapsed = time.time() - start_time
            
            # Calcular tokens
            output_length = len(generated_tokens)
            total_tokens = input_length + output_length
            
            return {
                "content": response.strip(),
                "usage": {
                    "prompt_tokens": input_length,
                    "completion_tokens": output_length,
                    "total_tokens": total_tokens,
                },
                "elapsed_seconds": round(elapsed, 2)
            }
            
        except Exception as e:
            print(f"[TRANSFORMERS ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Convierte mensajes a texto simple (fallback)
        
        Args:
            messages: Lista de mensajes
            
        Returns:
            Texto concatenado
        """
        text_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                text_parts.append(f"System: {content}\n")
            elif role == "user":
                text_parts.append(f"User: {content}\n")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}\n")
        
        text_parts.append("Assistant:")
        return "\n".join(text_parts)
    
    def unload_model(self):
        """Descarga el modelo de memoria"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_loaded = False
        
        # Limpiar caché de GPU si está disponible
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        print("[TRANSFORMERS] Model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo"""
        info = {
            "backend": "Transformers",
            "model_name_or_path": self.model_name_or_path,
            "model_type": self.model_type,
            "supports_native_system": self.supports_native_system,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "is_loaded": self._is_loaded,
        }
        
        if self._is_loaded and self.model:
            # Información adicional del modelo cargado
            if hasattr(self.model, 'config'):
                config = self.model.config
                info["architecture"] = config.architectures[0] if hasattr(config, 'architectures') else "unknown"
                info["vocab_size"] = getattr(config, 'vocab_size', None)
                info["hidden_size"] = getattr(config, 'hidden_size', None)
                info["num_layers"] = getattr(config, 'num_hidden_layers', None)
        
        return info
    
    @property
    def is_loaded(self) -> bool:
        """Indica si el modelo está cargado"""
        return self._is_loaded

