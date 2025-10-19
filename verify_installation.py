#!/usr/bin/env python3
"""
Script de verificación para la instalación de local-llm-chat
Verifica que todos los componentes funcionen correctamente
"""

import sys

def test_imports():
    """Prueba que todos los módulos puedan ser importados"""
    print("Testing imports...")

    try:
        from local_llm_chat import UniversalChatClient
        print("  ✓ UniversalChatClient")
    except ImportError as e:
        print(f"  ✗ UniversalChatClient: {e}")
        return False

    try:
        from local_llm_chat.model_config import detect_model_type, get_chat_format
        print("  ✓ model_config")
    except ImportError as e:
        print(f"  ✗ model_config: {e}")
        return False

    try:
        from local_llm_chat.cli import run_cli
        print("  ✓ cli")
    except ImportError as e:
        print(f"  ✗ cli: {e}")
        return False

    try:
        from local_llm_chat.utils import list_local_models
        print("  ✓ utils")
    except ImportError as e:
        print(f"  ✗ utils: {e}")
        return False

    return True


def test_model_detection():
    """Prueba las funciones de detección de modelos"""
    print("\nTesting model detection...")

    from local_llm_chat.model_config import detect_model_type, get_chat_format

    test_cases = [
        ("llama-3-8b-instruct.gguf", "llama-3", "llama-3"),
        ("gemma-2-9b-it-Q8_0.gguf", "gemma", "gemma"),
        ("mistral-7b-instruct-v0.2.gguf", "mistral", "mistral"),
    ]

    for filename, expected_type, expected_format in test_cases:
        detected_type = detect_model_type(filename)
        detected_format = get_chat_format(detected_type)

        if detected_type == expected_type and detected_format == expected_format:
            print(f"  ✓ {filename}: {detected_type} / {detected_format}")
        else:
            print(f"  ✗ {filename}: expected {expected_type}/{expected_format}, got {detected_type}/{detected_format}")
            return False

    return True


def test_local_models():
    """Prueba el descubrimiento de modelos locales"""
    print("\nTesting local model discovery...")

    from local_llm_chat.utils import list_local_models

    models = list_local_models()
    print(f"  Found {len(models)} local model(s)")

    if models:
        for model in models:
            print(f"    - {model}")
    else:
        print("    (No models found - this is OK for fresh installations)")

    return True


def test_package_metadata():
    """Prueba los metadatos del paquete"""
    print("\nTesting package metadata...")

    try:
        from local_llm_chat import __version__, __author__
        print(f"  ✓ Version: {__version__}")
        print(f"  ✓ Author: {__author__}")
        return True
    except ImportError as e:
        print(f"  ✗ Metadata import failed: {e}")
        return False


def main():
    """Ejecuta todas las pruebas de verificación"""
    print("=" * 60)
    print("LOCAL-LLM-CHAT INSTALLATION VERIFICATION")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Model Detection", test_model_detection),
        ("Local Models", test_local_models),
        ("Package Metadata", test_package_metadata),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))

    # Resumen
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Installation is working correctly.")
        print("\nYou can now run:")
        print("  python main.py")
        print("  python -m local_llm_chat")
        print("  local-llm-chat  (if installed)")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
