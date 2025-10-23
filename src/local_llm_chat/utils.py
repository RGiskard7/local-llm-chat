"""
Funciones de utilidad para descubrimiento y gestión de modelos
"""

import os
from .model_config import (
    get_hardware_info,
    get_smart_recommendations,
    estimate_model_size,
    POPULAR_MODELS
)

def list_local_models():
    """Lista todos los modelos GGUF en el directorio ./models."""
    models_dir = "./models"
    if not os.path.exists(models_dir):
        return []

    gguf_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.gguf'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ".")
                gguf_files.append(rel_path)

    return sorted(gguf_files)

def show_available_models():
    """Muestra los modelos locales y recomendaciones inteligentes."""
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
        for i, model in enumerate(local, 1):
            model_name = os.path.basename(model)
            size = estimate_model_size(model_name)
            print(f"  {i}. {model_name} (~{size}GB)")
            print(f"     Use: /changemodel {model}")
    else:
        print("\nLOCAL MODELS: None found")

    # Recomendaciones inteligentes
    print("\nRECOMMENDED MODELS (based on your hardware):")
    recommendations = get_smart_recommendations(timeout=10)

    if recommendations:
        for i, model in enumerate(recommendations[:5], 1):  # Top 5
            print(f"  {i}. {model['repo_id']}")
            print(f"     Size: ~{model['estimated_size_gb']}GB | Type: {model['model_type']}")
            print(f"     Downloads: {model['downloads']:,}")
            print(f"     Use: /download {i}")
    else:
        print("  [INFO] Could not fetch recommendations (using fallback)")
        print("\nFALLBACK MODELS:")
        for key, info in POPULAR_MODELS.items():
            print(f"  [{key}] {info['description']}")

    print("\n" + "=" * 60)
    print("USAGE:")
    print("  Change local: /changemodel <path>")
    print("  Download recommended: /download <number>")
    print("  Download fallback: Restart with model_key=<key>")
    print("=" * 60 + "\n")

    # Retornar recomendaciones para usar en el comando download
    return recommendations if recommendations else []

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
    print("  /download <num>  Download recommended model by number")
    print("  /changemodel <path>  Switch to different model")
    print("\nSYSTEM PROMPT:")
    print("  /system <text>   Set custom system prompt")
    print("  /showsystem      Display current prompt")
    print("  /clearsystem     Remove system prompt")
    print("  /preset <name>   Load preset from prompts.py")
    print("  /presets         List available presets")
    print("\nRAG MODE (Document Q&A):")
    print("  /rag <file>      Process document for RAG queries")
    print("  /ragstatus       Show current RAG document")
    print("  /disablerag      Disable RAG (return to normal chat)")
    print("  (After /rag, all messages use RAG context)")
    print("=" * 60 + "\n")
