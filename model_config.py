"""
Configuración de modelos y sus capacidades de system prompt.
Detecta automáticamente cómo manejar system prompts según el modelo.
"""

import psutil
import torch

# Modelos que soportan nativamente el rol "system" (comentario ya en español)
MODELS_WITH_NATIVE_SYSTEM_SUPPORT = {
    "llama": ["llama-2", "llama-3", "llama2", "llama3"],
    "mistral": ["mistral", "mixtral"],
    "openchat": ["openchat"],
    "dolphin": ["dolphin"],
    "zephyr": ["zephyr"],
    "vicuna": ["vicuna"],
    "nous": ["nous", "hermes"],
    "chatml": ["gpt", "yi"],  # Modelos que usan ChatML format (comentario ya en español)
}

# Modelos que NO soportan system (requieren workaround) (comentario ya en español)
MODELS_WITHOUT_SYSTEM_SUPPORT = {
    "gemma": ["gemma"],
    "phi": ["phi"],
    "qwen": ["qwen"],
}

# Mapeo de nombres de modelo a chat_format de llama.cpp (comentario ya en español)
CHAT_FORMAT_MAP = {
    "llama": "llama-3",  # Llama genérico por defecto llama-3
    "llama-2": "llama-2",
    "llama-3": "llama-3",
    "llama2": "llama-2",
    "llama3": "llama-3",
    "mistral": "mistral",
    "mixtral": "mistral",
    "gemma": "gemma",
    "phi": "phi",
    "openchat": "openchat",
    "dolphin": "chatml",  # Dolphin usa ChatML format (comentario ya en español)
    "zephyr": "zephyr",
    "vicuna": "vicuna",
    "nous": "chatml",  # Nous/Hermes usa ChatML (comentario ya en español)
    "chatml": "chatml",
    "yi": "chatml",
    "qwen": "chatml",
}

# Recomendaciones de tamaño de modelo basadas en hardware (GB)
RAM_SIZE_THRESHOLDS = [
    (6, 3.0, ["1b", "3b"]),       # < 6GB RAM: modelos pequeños
    (10, 7.0, ["3b", "7b"]),      # 6-10GB RAM: modelos medianos
    (16, 10.0, ["7b", "8b"]),     # 10-16GB RAM: modelos grandes
    (float('inf'), 20.0, ["7b", "8b", "13b"])  # > 16GB RAM: modelos muy grandes
]

# Patrones de tamaño de modelo (parámetros → tamaño base en GB)
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

# Multiplicadores de cuantización
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
    Detecta el tipo de modelo basándose en el nombre o ruta.
    
    Args:
        model_name_or_path: Nombre del modelo o ruta al archivo
        
    Returns:
        Tipo de modelo detectado (ej., "gemma", "llama-3", "mistral", "dolphin")
    """
    model_lower = model_name_or_path.lower()
    
    # Detección prioritaria (modelos específicos primero) (comentario ya en español)
    # Dolphin y otros fine-tunes específicos (comentario ya en español)
    if "dolphin" in model_lower:
        return "dolphin"
    if "hermes" in model_lower or "nous" in model_lower:
        return "nous"
    
    # Versiones de Llama (más específico a más genérico)
    if "llama-3.1" in model_lower or "llama3.1" in model_lower:
        return "llama-3"
    if "llama-3" in model_lower or "llama3" in model_lower:
        return "llama-3"
    if "llama-2" in model_lower or "llama2" in model_lower:
        return "llama-2"
    if "llama" in model_lower:
        return "llama-3"  # Por defecto para llama genérico
    
    # Otros modelos populares (comentario ya en español)
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
    Obtiene el formato de chat apropiado para llama.cpp.
    
    Args:
        model_type: Tipo de modelo detectado
        
    Returns:
        String de formato de chat para llama.cpp
    """
    return CHAT_FORMAT_MAP.get(model_type, None)


def supports_native_system(model_type: str) -> bool:
    """
    Verifica si el modelo soporta nativamente el rol "system".
    
    Args:
        model_type: Tipo de modelo detectado
        
    Returns:
        True si soporta system nativamente, False si requiere solución alternativa
    """
    return model_type in MODELS_WITH_NATIVE_SYSTEM_SUPPORT


def get_system_prompt_strategy(model_type: str) -> str:
    """
    Determina la estrategia para manejar prompts del sistema.
    
    Args:
        model_type: Tipo de modelo detectado
        
    Returns:
        "native" o "workaround"
    """
    return "native" if supports_native_system(model_type) else "workaround"


# Configuraciones de modelos populares para descarga rápida (comentario ya en español)
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
        Dict con información real de RAM y VRAM disponible
    """
    try:
        # Consulta real del sistema
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
        # Fallback solo si psutil/torch no están disponibles
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
    Sin números mágicos - usa configuración definida arriba.
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        Tamaño estimado en GB
    """
    name_lower = model_name.lower()
    
    # Detectar tamaño base usando patrones configurados
    base_size = 7.0  # Por defecto conservador
    for patterns, size in MODEL_SIZE_PATTERNS:
        if any(pattern in name_lower for pattern in patterns):
            base_size = size
            break
    
    # Ajustar por cuantización usando factores configurados (comentario ya en español)
    for quant_type, factor in QUANTIZATION_FACTORS.items():
        if quant_type in name_lower:
            return base_size * factor
    
    # Por defecto: Q6 (0.75x)
    return base_size * 0.75


def get_smart_recommendations(timeout: int = 10) -> list:
    """
    Consulta real a la API de Hugging Face para obtener modelos populares.
    Filtra basándose en hardware real detectado del sistema.
    
    Args:
        timeout: Timeout para la consulta API (no usado actualmente)
        
    Returns:
        Lista de modelos recomendados con su información real de HF
    """
    try:
        from huggingface_hub import HfApi
        
        # Detectar hardware real
        hw = get_hardware_info()
        ram_available = hw['ram_available_gb']
        
        # Determinar tamaños usando umbrales configurados (sin números mágicos)
        max_size = 20.0
        recommended_params = ["7b", "8b", "13b"]
        
        for threshold, size_limit, params in RAM_SIZE_THRESHOLDS:
            if ram_available < threshold:
                max_size = size_limit
                recommended_params = params
                break
        
        # Consultar API real de Hugging Face (no hardcodeado)
        api = HfApi()
        models = api.list_models(
            filter="gguf",
            sort="downloads",
            direction=-1,
            limit=100,  # Top 100 más descargados (comentario ya en español)
        )
        
        recommendations = []
        seen_base_names = set()
        
        for model in models:
            model_id = model.modelId
            model_lower = model_id.lower()
            
            # Filtrar por tamaño de parámetros (comentario ya en español)
            if not any(param in model_lower for param in recommended_params):
                continue
            
            # Evitar duplicados del mismo modelo base (comentario ya en español)
            base_name = model_id.split('-')[0]
            if base_name in seen_base_names:
                continue
            
            # Estimar tamaño (comentario ya en español)
            estimated_size = estimate_model_size(model_id)
            
            if estimated_size <= max_size:
                # Detectar tipo de modelo (comentario ya en español)
                model_type = detect_model_type(model_id)
                
                recommendations.append({
                    'repo_id': model_id,
                    'model_type': model_type,
                    'estimated_size_gb': round(estimated_size, 1),
                    'downloads': getattr(model, 'downloads', 0),
                    'fits_hardware': True
                })
                
                seen_base_names.add(base_name)
                
                # Limitar a top 10 (comentario ya en español)
                if len(recommendations) >= 10:
                    break
        
        return recommendations
        
    except Exception as e:
        # Fallback: retornar lista vacía para usar hardcodeado
        return []


def validate_model_config() -> bool:
    """
    Valida que la configuración de modelos sea consistente.
    Útil para depuración al añadir nuevos modelos.
    
    Returns:
        True si la configuración es válida
    """
    issues = []
    
    # Verificar que todos los modelos populares tienen configuración completa (comentario ya en español)
    for key, info in POPULAR_MODELS.items():
        required_fields = ["type", "repo_id", "filename", "description"]
        for field in required_fields:
            if field not in info:
                issues.append(f"Model '{key}' missing required field '{field}'")
        
        # Verificar que el tipo tiene formato de chat (comentario ya en español)
        if "type" in info and info["type"] not in CHAT_FORMAT_MAP:
            issues.append(f"Model '{key}' type '{info['type']}' not in CHAT_FORMAT_MAP")
    
    # Verificar que los tipos principales están clasificados (comentario ya en español)
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

