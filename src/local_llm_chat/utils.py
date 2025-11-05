"""
Funciones de utilidad para descubrimiento y gestión de modelos
"""

import os
from .model_config import (
    get_hardware_info,
    get_smart_recommendations,
    get_transformers_recommendations,
    estimate_model_size,
)

def list_local_models():
    """
    Lista todos los modelos locales (GGUF y Transformers) en ./models.
    
    Returns:
        Lista de diccionarios con información de cada modelo:
        - 'path': ruta relativa al modelo
        - 'type': 'gguf' o 'transformers'
        - 'name': nombre del modelo
    """
    models_dir = "./models"
    if not os.path.exists(models_dir):
        return []

    models = []
    
    # Buscar modelos GGUF (archivos .gguf)
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.gguf'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ".")
                models.append({
                    'path': rel_path,
                    'type': 'gguf',
                    'name': os.path.basename(file)
                })
    
    # Buscar modelos Transformers (carpetas con config.json y model.safetensors/model.bin)
    for root, dirs, files in os.walk(models_dir):
        # Solo considerar carpetas que no sean la raíz
        if root == models_dir:
            continue
            
        # Verificar si es un modelo Transformers
        has_config = 'config.json' in files
        has_model = any(f.startswith('model.') and (f.endswith('.safetensors') or f.endswith('.bin')) for f in files)
        
        if has_config and has_model:
            # Es una carpeta de modelo Transformers
            rel_path = os.path.relpath(root, ".")
            models.append({
                'path': rel_path,
                'type': 'transformers',
                'name': os.path.basename(root)
            })
    
    # Ordenar por nombre
    models.sort(key=lambda x: x['name'])
    return models

def show_available_models():
    """Muestra los modelos locales y recomendaciones inteligentes para GGUF y Transformers."""
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)

    # Información del hardware
    hw = get_hardware_info()
    print(f"\nHARDWARE:")
    print(f"  RAM: {hw['ram_available_gb']:.1f}GB available / {hw['ram_total_gb']:.1f}GB total")
    if hw['has_gpu']:
        print(f"  VRAM: {hw['vram_available_gb']:.1f}GB available / {hw['vram_total_gb']:.1f}GB total (GPU)")
    else:
        print(f"  GPU: Not available (CPU only)")

    # Modelos locales
    local = list_local_models()
    if local:
        print("\nLOCAL MODELS:")
        for i, model_info in enumerate(local, 1):
            model_name = model_info['name']
            model_type = model_info['type'].upper()
            model_path = model_info['path']
            
            # Estimar tamaño
            if model_info['type'] == 'gguf':
                size = estimate_model_size(model_name)
            else:
                # Para Transformers, estimar desde el nombre o usar tamaño por defecto
                size = estimate_model_size(model_name)
            
            print(f"  {i}. {model_name} (~{size}GB) [{model_type}]")
            print(f"     Use: /changemodel {model_path}")
    else:
        print("\nLOCAL MODELS: None found")

    # Recomendaciones GGUF
    print("\n" + "=" * 60)
    print("GGUF MODELS (Recommended - Fast on CPU)")
    print("=" * 60)
    gguf_recommendations = get_smart_recommendations(timeout=10)

    if gguf_recommendations:
        for i, model in enumerate(gguf_recommendations[:5], 1):  # Top 5
            print(f"  {i}. {model['repo_id']}")
            print(f"     Size: ~{model['estimated_size_gb']}GB | Type: {model['model_type']}")
            print(f"     Downloads: {model['downloads']:,}")
            print(f"     Use: /download {i}")
    else:
        print("  [INFO] Could not fetch GGUF recommendations")

    # Recomendaciones Transformers
    print("\n" + "=" * 60)
    print("TRANSFORMERS MODELS (More RAM, any HF model)")
    print("=" * 60)
    transformers_recommendations = get_transformers_recommendations(timeout=10)

    if transformers_recommendations:
        for i, model in enumerate(transformers_recommendations[:5], 1):  # Top 5
            idx = len(gguf_recommendations) + i  # Continuar numeración
            print(f"  {idx}. {model['repo_id']}")
            print(f"     Size: ~{model['estimated_size_gb']}GB | Type: {model['model_type']}")
            print(f"     Downloads: {model['downloads']:,}")
            print(f"     Use: /download {idx}")
    else:
        print("  [INFO] Could not fetch Transformers recommendations")

    print("\n" + "=" * 60)
    print("USAGE:")
    print("  Change local: /changemodel <path>")
    print("  Download recommended: /download <number>")
    print("  GGUF = Fast on CPU, lower RAM usage")
    print("  Transformers = More models available, higher RAM")
    print("=" * 60 + "\n")

    # Retornar recomendaciones combinadas para usar en el comando download
    all_recommendations = gguf_recommendations + transformers_recommendations
    return all_recommendations if all_recommendations else []

def show_help():
    """Muestra la referencia de comandos."""
    print("\n" + "=" * 60)
    print("  COMMAND REFERENCE")
    print("=" * 60)
    print("\nCONVERSATION:")
    print("  /exit            Exit chat and save log")
    print("  /save            Save conversation")
    print("  /history         Show full history")
    print("  /clear           Clear history")
    print("  /stats           Show session statistics")
    print("  /help            Show this help menu")
    print("\nMODEL MANAGEMENT:")
    print("  /models          List available models + smart recommendations")
    print("  /download <num|id>  Download model by number or HuggingFace ID")
    print("                   Examples: /download 1")
    print("                             /download meta-llama/Llama-3.1-8B-GGUF")
    print("  /changemodel <path>  Switch to different model")
    print("\nSYSTEM PROMPT:")
    print("  /system <text>   Set custom system prompt")
    print("  /showsystem      Display current prompt")
    print("  /clearsystem     Remove system prompt")
    print("  /preset <name>   Load preset from prompts.py")
    print("  /presets         List available presets")
    print("\nRAG MODE (Document Q&A - Professional Commands):")
    print("  /load <file>     Load document into RAG system")
    print("  /unload <file>   Remove document from RAG system")
    print("  /list            List all loaded documents")
    print("  /clear           Clear all documents from RAG")
    print("  /rag on          Activate RAG mode (search in documents)")
    print("  /rag off         Deactivate RAG mode (use base knowledge)")
    print("  /status          Show RAG status and loaded documents")
    print("\nRAG WORKFLOW:")
    print("  1. /load doc.pdf       # Load document")
    print("  2. /rag on             # Activate RAG")
    print("  3. Ask questions...    # RAG searches in documents")
    print("  4. /rag off            # Deactivate RAG (chat freely)")
    print("  5. /rag on             # Reactivate when needed")
    print("=" * 60 + "\n")
