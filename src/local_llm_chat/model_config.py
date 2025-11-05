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
# GGUF models (cuantizados, menos RAM)
RAM_SIZE_THRESHOLDS = [
    (6, 3.0, ["1b", "3b"]),       # < 6GB RAM: small models
    (10, 7.0, ["3b", "7b"]),      # 6-10GB RAM: medium models
    (16, 10.0, ["7b", "8b"]),     # 10-16GB RAM: large models
    (float('inf'), 20.0, ["7b", "8b", "13b"])  # > 16GB RAM: very large models
]

# Transformers models (full precision, más RAM)
TRANSFORMERS_RAM_THRESHOLDS = [
    (8, 2.0, ["0.5b", "500m", "560m", "0.6b"]),    # < 8GB RAM: tiny models
    (16, 4.0, ["1b", "1.5b"]),                      # 8-16GB RAM: small models
    (32, 8.0, ["3b", "7b"]),                        # 16-32GB RAM: medium models
    (float('inf'), 16.0, ["7b", "8b"])              # > 32GB RAM: large models
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

# Full precision size multiplier (Transformers sin cuantización)
# Un modelo de 7B parámetros en float32/float16 ocupa ~2x el tamaño base
FULL_PRECISION_SIZE_MULTIPLIER = 2.0

# Preferencias de cuantización GGUF (orden de preferencia)
GGUF_QUANTIZATION_PREFERENCES = ['q8_0', 'q8', 'q6', 'q5', 'q4']

# Conversión de bytes a GB
BYTES_TO_GB = 1024 ** 3

# Máximo número de shards a considerar para modelos Transformers (algunos modelos grandes tienen múltiples archivos)
MAX_TRANSFORMERS_SHARDS = 5


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


def _select_preferred_gguf_file(gguf_files: list) -> str:
    """
    Selecciona el archivo GGUF preferido según orden de calidad.
    
    Args:
        gguf_files: Lista de nombres de archivos GGUF
        
    Returns:
        Nombre del archivo preferido, o el primero si no hay coincidencias
    """
    if not gguf_files:
        return None
    
    # Buscar según preferencias de cuantización
    for preference in GGUF_QUANTIZATION_PREFERENCES:
        for f in gguf_files:
            if preference in f.lower():
                return f
    
    # Si no hay coincidencias, usar el primero
    return gguf_files[0]


def _get_gguf_file_size(api, repo_id: str) -> float:
    """
    Obtiene el tamaño REAL del archivo GGUF que se descargará.
    
    Args:
        api: Instancia de HfApi
        repo_id: ID del repositorio
        
    Returns:
        Tamaño en GB del archivo GGUF que se descargará, o 0 si no se puede determinar
    """
    try:
        # Listar archivos del repo
        files = api.list_repo_files(repo_id, repo_type="model")
        gguf_files = [f for f in files if f.endswith('.gguf')]
        
        if not gguf_files:
            return 0
        
        # Elegir el archivo preferido usando función reutilizable
        preferred_file = _select_preferred_gguf_file(gguf_files)
        if not preferred_file:
            return estimate_model_size(repo_id)
        
        # Intentar obtener tamaño del archivo usando repo_file_info (si existe)
        try:
            if hasattr(api, 'repo_file_info'):
                file_info = api.repo_file_info(repo_id, preferred_file, repo_type="model")
                if file_info and hasattr(file_info, 'size') and file_info.size:
                    return file_info.size / BYTES_TO_GB
        except (AttributeError, Exception):
            # Si repo_file_info no existe o falla, continuar con fallback
            pass
        
        # Fallback: estimar desde el nombre del archivo (más preciso que desde repo)
        return estimate_model_size(preferred_file)
        
    except Exception:
        # Si falla, usar estimación desde nombre del repo
        return estimate_model_size(repo_id)


def get_smart_recommendations(timeout: int = 10) -> list:
    """
    Consulta REAL la API de Hugging Face para obtener modelos GGUF populares.
    Filtra basándose en hardware REAL detectado del sistema.
    Obtiene el tamaño REAL del archivo que se descargará.
    
    Args:
        timeout: Timeout para la consulta API (no usado actualmente)
        
    Returns:
        Lista de modelos GGUF recomendados con su información REAL de HF
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
            
            # Obtener tamaño REAL del archivo que se descargará
            real_size = _get_gguf_file_size(api, model_id)
            
            # Si no se pudo obtener tamaño real, usar estimación como fallback
            if real_size == 0:
                real_size = estimate_model_size(model_id)
            
            if real_size <= max_size:
                # Detectar tipo de modelo
                model_type = detect_model_type(model_id)
                
                recommendations.append({
                    'repo_id': model_id,
                    'model_type': model_type,
                    'estimated_size_gb': round(real_size, 1),
                    'downloads': getattr(model, 'downloads', 0),
                    'fits_hardware': True,
                    'backend': 'gguf'
                })
                
                seen_base_names.add(base_name)
                
                # Limitar a top 10
                if len(recommendations) >= 10:
                    break
        
        return recommendations
        
    except Exception as e:
        # Fallback: retornar lista vacía si falla la consulta API
        return []


def _get_transformers_model_size(api, repo_id: str) -> float:
    """
    Obtiene el tamaño REAL aproximado del modelo Transformers.
    Suma el tamaño de los archivos principales (model.safetensors o model.bin).
    
    Args:
        api: Instancia de HfApi
        repo_id: ID del repositorio
        
    Returns:
        Tamaño aproximado en GB, o 0 si no se puede determinar
    """
    try:
        # Listar archivos del repo
        files = api.list_repo_files(repo_id, repo_type="model")
        
        # Buscar archivos del modelo principal
        model_files = [f for f in files if f.startswith('model.') and (f.endswith('.safetensors') or f.endswith('.bin'))]
        
        if not model_files:
            # Fallback: estimar desde nombre
            return estimate_model_size(repo_id) * FULL_PRECISION_SIZE_MULTIPLIER
        
        # Obtener tamaño de los archivos del modelo
        total_size_bytes = 0
        file_paths = model_files[:MAX_TRANSFORMERS_SHARDS]
        
        # Intentar obtener tamaño usando repo_file_info (si existe)
        if hasattr(api, 'repo_file_info'):
            for file_path in file_paths:
                try:
                    file_info = api.repo_file_info(repo_id, file_path, repo_type="model")
                    if file_info and hasattr(file_info, 'size') and file_info.size:
                        total_size_bytes += file_info.size
                except Exception:
                    # Si falla, continuar con siguiente archivo
                    continue
        
        if total_size_bytes > 0:
            return total_size_bytes / BYTES_TO_GB
        
        # Fallback: estimar desde nombre del modelo
        return estimate_model_size(repo_id) * FULL_PRECISION_SIZE_MULTIPLIER
        
    except Exception:
        # Si falla completamente, usar estimación desde nombre
        return estimate_model_size(repo_id) * FULL_PRECISION_SIZE_MULTIPLIER


def get_transformers_recommendations(timeout: int = 10) -> list:
    """
    Consulta la API de Hugging Face para obtener modelos Transformers populares.
    Filtra basándose en hardware REAL detectado (Transformers necesita más RAM).
    
    Args:
        timeout: Timeout para la consulta API
        
    Returns:
        Lista de modelos Transformers recomendados
    """
    try:
        from huggingface_hub import HfApi
        
        # Detectar hardware REAL
        hw = get_hardware_info()
        ram_available = hw['ram_available_gb']
        
        # Determinar tamaños usando thresholds de Transformers
        max_size = 16.0
        recommended_params = ["7b", "8b"]
        
        for threshold, size_limit, params in TRANSFORMERS_RAM_THRESHOLDS:
            if ram_available < threshold:
                max_size = size_limit
                recommended_params = params
                break
        
        # Consultar API de Hugging Face
        api = HfApi()
        models = api.list_models(
            filter="text-generation-inference",
            sort="downloads",
            direction=-1,
            limit=200,  # Más porque hay mucha variedad
        )
        
        recommendations = []
        seen_base_names = set()
        
        for model in models:
            model_id = model.modelId
            model_lower = model_id.lower()
            
            # Saltar modelos GGUF (ya están en otra lista)
            if "gguf" in model_lower:
                continue
            
            # Filtrar por tamaño de parámetros
            if not any(param in model_lower for param in recommended_params):
                continue
            
            # Evitar duplicados del mismo modelo base
            base_name = model_id.split('/')[1] if '/' in model_id else model_id
            base_name = base_name.split('-')[0]
            if base_name in seen_base_names:
                continue
            
            # Obtener tamaño REAL del modelo (o estimar desde nombre)
            estimated_size = _get_transformers_model_size(api, model_id)
            
            # Si no se pudo obtener tamaño real, usar estimación desde nombre
            if estimated_size == 0:
                for patterns, base_gb in MODEL_SIZE_PATTERNS:
                    if any(p in model_lower for p in patterns):
                        estimated_size = base_gb * FULL_PRECISION_SIZE_MULTIPLIER
                        break
            
            # Si aún no se tiene tamaño, usar estimación conservadora
            if estimated_size == 0:
                estimated_size = 7.0 * FULL_PRECISION_SIZE_MULTIPLIER  # Default conservador
            
            if estimated_size > max_size:
                continue
            
            # Detectar tipo de modelo
            model_type = detect_model_type(model_id)
            
            recommendations.append({
                'repo_id': model_id,
                'model_type': model_type,
                'estimated_size_gb': round(estimated_size, 1),
                'downloads': getattr(model, 'downloads', 0),
                'fits_hardware': True,
                'backend': 'transformers'
            })
            
            seen_base_names.add(base_name)
            
            # Limitar a top 10
            if len(recommendations) >= 10:
                break
        
        # Ordenar por downloads (métrica objetiva)
        recommendations.sort(key=lambda x: -x['downloads'])
        
        return recommendations[:10]
        
    except Exception as e:
        # Fallback: retornar lista vacía
        return []


def validate_model_config() -> bool:
    """
    Valida que la configuración de modelos sea consistente.
    Útil para debugging al añadir nuevos modelos.
    
    Returns:
        True si la configuración es válida
    """
    issues = []
    
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
        model_identifier: Path local o nombre HF (repo_id)
        
    Returns:
        "gguf" o "transformers"
    """
    # Si es un path y termina en .gguf
    if os.path.exists(model_identifier) and model_identifier.endswith('.gguf'):
        return "gguf"
    
    # Si contiene .gguf o GGUF en el nombre (incluso si no existe aún)
    model_lower = model_identifier.lower()
    if '.gguf' in model_lower or 'gguf' in model_lower:
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
    
    # Default: asumir GGUF para compatibilidad con código existente
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

