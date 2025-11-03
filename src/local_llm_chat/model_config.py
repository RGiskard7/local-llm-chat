"""
Configuración de modelos y sus capacidades de system prompt.
Detecta automáticamente cómo manejar system prompts según el modelo.
Soporta detección de tipo de backend (GGUF vs Transformers).
"""

import os
import psutil
import torch

# Modelos que soportan nativamente el rol "system"
MODELS_WITH_NATIVE_SYSTEM_SUPPORT = {
    "llama": ["llama-2", "llama-3", "llama2", "llama3"],
    "mistral": ["mistral", "mixtral"],
    "openchat": ["openchat"],
    "dolphin": ["dolphin"],
    "zephyr": ["zephyr"],
    "vicuna": ["vicuna"],
    "nous": ["nous", "hermes"],
    "chatml": ["gpt", "yi"],  # Modelos que usan ChatML format
}

# Modelos que NO soportan system (requieren workaround)
MODELS_WITHOUT_SYSTEM_SUPPORT = {
    "gemma": ["gemma"],
    "phi": ["phi"],
    "qwen": ["qwen"],
}

# Mapeo de nombres de modelo a chat_format de llama.cpp
CHAT_FORMAT_MAP = {
    "llama": "llama-3",  # Generic llama defaults to llama-3
    "llama-2": "llama-2",
    "llama-3": "llama-3",
    "llama2": "llama-2",
    "llama3": "llama-3",
    "mistral": "mistral",
    "mixtral": "mistral",
    "gemma": "gemma",
    "phi": "phi",
    "openchat": "openchat",
    "dolphin": "chatml",  # Dolphin usa ChatML format
    "zephyr": "zephyr",
    "vicuna": "vicuna",
    "nous": "chatml",  # Nous/Hermes usa ChatML
    "chatml": "chatml",
    "yi": "chatml",
    "qwen": "chatml",
}

# Hardware-based model size recommendations (GB)
RAM_SIZE_THRESHOLDS = [
    (6, 3.0, ["1b", "3b"]),       # < 6GB RAM: small models
    (10, 7.0, ["3b", "7b"]),      # 6-10GB RAM: medium models
    (16, 10.0, ["7b", "8b"]),     # 10-16GB RAM: large models
    (float('inf'), 20.0, ["7b", "8b", "13b"])  # > 16GB RAM: very large models
]

# Model size patterns (parameters → base size in GB)
MODEL_SIZE_PATTERNS = [
    (["0.5b", "500m"], 0.5),
    (["1b", "1.5b"], 1.5),
    (["3b"], 3.0),
    (["7b"], 7.0),
    (["8b"], 8.0),
    (["13b"], 13.0),
    (["34b"], 20.0),
    (["70b"], 40.0),
]

# Quantization multipliers
QUANTIZATION_FACTORS = {
    "q4": 0.5,
    "q4_k": 0.5,
    "q5": 0.625,
    "q5_k": 0.625,
    "q6": 0.75,
    "q6_k": 0.75,
    "q8": 1.0,
}


def detect_model_type(model_name_or_path: str) -> str:
    """
    Detecta el tipo de modelo basándose en el nombre o path.
    
    Args:
        model_name_or_path: Nombre del modelo o path al archivo
        
    Returns:
        Tipo de modelo detectado (e.g., "gemma", "llama-3", "mistral", "dolphin")
    """
    model_lower = model_name_or_path.lower()
    
    # Detección prioritaria (modelos específicos primero)
    # Dolphin y otros fine-tunes específicos
    if "dolphin" in model_lower:
        return "dolphin"
    if "hermes" in model_lower or "nous" in model_lower:
        return "nous"
    
    # Llama versions (más específico a más genérico)
    if "llama-3.1" in model_lower or "llama3.1" in model_lower:
        return "llama-3"
    if "llama-3" in model_lower or "llama3" in model_lower:
        return "llama-3"
    if "llama-2" in model_lower or "llama2" in model_lower:
        return "llama-2"
    if "llama" in model_lower:
        return "llama-3"  # Default para llama genérico
    
    # Otros modelos populares
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    if "gemma" in model_lower:
        return "gemma"
    if "phi-3" in model_lower or "phi3" in model_lower:
        return "phi"
    if "openchat" in model_lower:
        return "openchat"
    if "zephyr" in model_lower:
        return "zephyr"
    if "vicuna" in model_lower:
        return "vicuna"
    if "yi" in model_lower:
        return "yi"
    if "qwen" in model_lower:
        return "qwen"
    
    return "unknown"


def get_chat_format(model_type: str) -> str:
    """
    Obtiene el chat format apropiado para llama.cpp.
    
    Args:
        model_type: Tipo de modelo detectado
        
    Returns:
        Chat format string para llama.cpp
    """
    return CHAT_FORMAT_MAP.get(model_type, None)


def supports_native_system(model_type: str) -> bool:
    """
    Verifica si el modelo soporta nativamente el rol "system".
    
    Args:
        model_type: Tipo de modelo detectado
        
    Returns:
        True si soporta system nativamente, False si requiere workaround
    """
    return model_type in MODELS_WITH_NATIVE_SYSTEM_SUPPORT


def get_system_prompt_strategy(model_type: str) -> str:
    """
    Determina la estrategia para manejar system prompts.
    
    Args:
        model_type: Tipo de modelo detectado
        
    Returns:
        "native" o "workaround"
    """
    return "native" if supports_native_system(model_type) else "workaround"


# Configuraciones de modelos populares para descarga rápida
POPULAR_MODELS = {
    "gemma-12b": {
        "repo_id": "unsloth/gemma-3-12b-it-GGUF",
        "filename": "gemma-3-12b-it-UD-Q8_K_XL.gguf",
        "type": "gemma",
        "description": "Gemma 3 12B - Modelo conversacional de Google"
    },
    "llama-3-8b": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
        "type": "llama-3",
        "description": "Llama 3 8B Instruct - Modelo de Meta"
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q8_0.gguf",
        "type": "mistral",
        "description": "Mistral 7B Instruct - Modelo de Mistral AI"
    },
    "phi-3-mini": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "type": "phi",
        "description": "Phi-3 Mini - Modelo compacto de Microsoft"
    },
    "openchat-3.5": {
        "repo_id": "TheBloke/openchat-3.5-1210-GGUF",
        "filename": "openchat-3.5-1210.Q8_0.gguf",
        "type": "openchat",
        "description": "OpenChat 3.5 - Modelo conversacional optimizado"
    }
}


def get_model_info(model_key: str) -> dict:
    """
    Obtiene información de un modelo popular.
    
    Args:
        model_key: Clave del modelo en POPULAR_MODELS
        
    Returns:
        Diccionario con información del modelo
    """
    return POPULAR_MODELS.get(model_key)


def list_popular_models() -> list:
    """
    Lista todos los modelos populares disponibles.
    
    Returns:
        Lista de tuplas (key, description)
    """
    return [(key, info["description"]) for key, info in POPULAR_MODELS.items()]


def get_hardware_info() -> dict:
    """
    Detecta el hardware disponible del sistema en tiempo real.
    
    Returns:
        Dict con información REAL de RAM y VRAM disponible
    """
    try:
        # Consulta REAL del sistema
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_available = psutil.virtual_memory().available / (1024**3)
        
        vram_total = 0
        vram_available = 0
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_available = vram_total - (torch.cuda.memory_allocated(0) / (1024**3))
        
        return {
            'ram_total_gb': round(ram_total, 1),
            'ram_available_gb': round(ram_available, 1),
            'vram_total_gb': round(vram_total, 1),
            'vram_available_gb': round(vram_available, 1),
            'has_gpu': has_gpu
        }
    except Exception:
        # Fallback solo si psutil/torch no disponibles
        return {
            'ram_total_gb': 8.0,
            'ram_available_gb': 4.0,
            'vram_total_gb': 0.0,
            'vram_available_gb': 0.0,
            'has_gpu': False
        }


def estimate_model_size(model_name: str) -> float:
    """
    Estima el tamaño de un modelo basándose en su nombre.
    Sin magic numbers - usa configuración definida arriba.
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        Tamaño estimado en GB
    """
    name_lower = model_name.lower()
    
    # Detectar tamaño base usando patterns configurados
    base_size = 7.0  # Default conservador
    for patterns, size in MODEL_SIZE_PATTERNS:
        if any(pattern in name_lower for pattern in patterns):
            base_size = size
            break
    
    # Ajustar por cuantización usando factores configurados
    for quant_type, factor in QUANTIZATION_FACTORS.items():
        if quant_type in name_lower:
            return base_size * factor
    
    # Default: Q6 (0.75x)
    return base_size * 0.75


def get_smart_recommendations(timeout: int = 10) -> list:
    """
    Consulta REAL la API de Hugging Face para obtener modelos populares.
    Filtra basándose en hardware REAL detectado del sistema.
    
    Args:
        timeout: Timeout para la consulta API (no usado actualmente)
        
    Returns:
        Lista de modelos recomendados con su información REAL de HF
    """
    try:
        from huggingface_hub import HfApi
        
        # Detectar hardware REAL
        hw = get_hardware_info()
        ram_available = hw['ram_available_gb']
        
        # Determinar tamaños usando thresholds configurados (sin magic numbers)
        max_size = 20.0
        recommended_params = ["7b", "8b", "13b"]
        
        for threshold, size_limit, params in RAM_SIZE_THRESHOLDS:
            if ram_available < threshold:
                max_size = size_limit
                recommended_params = params
                break
        
        # Consultar API REAL de Hugging Face (no hardcoded)
        api = HfApi()
        models = api.list_models(
            filter="gguf",
            sort="downloads",
            direction=-1,
            limit=100,  # Top 100 más descargados
        )
        
        recommendations = []
        seen_base_names = set()
        
        for model in models:
            model_id = model.modelId
            model_lower = model_id.lower()
            
            # Filtrar por tamaño de parámetros
            if not any(param in model_lower for param in recommended_params):
                continue
            
            # Evitar duplicados del mismo modelo base
            base_name = model_id.split('-')[0]
            if base_name in seen_base_names:
                continue
            
            # Estimar tamaño
            estimated_size = estimate_model_size(model_id)
            
            if estimated_size <= max_size:
                # Detectar tipo de modelo
                model_type = detect_model_type(model_id)
                
                recommendations.append({
                    'repo_id': model_id,
                    'model_type': model_type,
                    'estimated_size_gb': round(estimated_size, 1),
                    'downloads': getattr(model, 'downloads', 0),
                    'fits_hardware': True
                })
                
                seen_base_names.add(base_name)
                
                # Limitar a top 10
                if len(recommendations) >= 10:
                    break
        
        return recommendations
        
    except Exception as e:
        # Fallback: retornar lista vacía para usar hardcoded
        return []


def validate_model_config() -> bool:
    """
    Valida que la configuración de modelos sea consistente.
    Útil para debugging al añadir nuevos modelos.
    
    Returns:
        True si la configuración es válida
    """
    issues = []
    
    # Verificar que todos los modelos populares tienen configuración completa
    for key, info in POPULAR_MODELS.items():
        required_fields = ["type", "repo_id", "filename", "description"]
        for field in required_fields:
            if field not in info:
                issues.append(f"Model '{key}' missing required field '{field}'")
        
        # Verificar que el tipo tiene chat format
        if "type" in info and info["type"] not in CHAT_FORMAT_MAP:
            issues.append(f"Model '{key}' type '{info['type']}' not in CHAT_FORMAT_MAP")
    
    # Verificar que los tipos principales están clasificados
    all_supported_types = set(MODELS_WITH_NATIVE_SYSTEM_SUPPORT.keys()) | set(MODELS_WITHOUT_SYSTEM_SUPPORT.keys())
    for model_type in all_supported_types:
        if model_type not in CHAT_FORMAT_MAP:
            issues.append(f"Model type '{model_type}' in support lists but missing from CHAT_FORMAT_MAP")
    
    if issues:
        print("[WARN] Model configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


def detect_backend_type(model_identifier: str) -> str:
    """
    Detecta el tipo de backend basándose en el identificador del modelo
    
    Args:
        model_identifier: Path local, nombre HF, o model_key
        
    Returns:
        "gguf" o "transformers"
    """
    # Si es un path y termina en .gguf
    if os.path.exists(model_identifier) and model_identifier.endswith('.gguf'):
        return "gguf"
    
    # Si contiene .gguf en el nombre (incluso si no existe aún)
    if '.gguf' in model_identifier.lower():
        return "gguf"
    
    # Si parece un nombre de HuggingFace (contiene /)
    if '/' in model_identifier:
        return "transformers"
    
    # Si es un path local que existe y no es GGUF
    if os.path.exists(model_identifier):
        # Verificar si tiene archivos de Transformers
        if os.path.isdir(model_identifier):
            has_config = os.path.exists(os.path.join(model_identifier, "config.json"))
            has_pytorch = any(
                f.endswith('.bin') or f.endswith('.safetensors')
                for f in os.listdir(model_identifier)
            )
            if has_config or has_pytorch:
                return "transformers"
    
    # Default: asumir GGUF para compatibilidad con código legacy
    return "gguf"


def is_gguf_model(model_identifier: str) -> bool:
    """
    Verifica si un modelo es GGUF
    
    Args:
        model_identifier: Path o nombre del modelo
        
    Returns:
        True si es GGUF
    """
    return detect_backend_type(model_identifier) == "gguf"


def is_transformers_model(model_identifier: str) -> bool:
    """
    Verifica si un modelo es Transformers
    
    Args:
        model_identifier: Path o nombre del modelo
        
    Returns:
        True si es Transformers
    """
    return detect_backend_type(model_identifier) == "transformers"

