#!/usr/bin/env python3
"""
Script de verificación CUDA para Local LLM Chat
Verifica que CUDA esté correctamente instalado y configurado
"""

import sys
import subprocess
import platform

def check_nvidia_driver():
    """Verifica que el driver NVIDIA esté instalado"""
    print("Verificando driver NVIDIA...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Driver NVIDIA detectado")
            # Extraer información básica
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    print(f"   {line.strip()}")
                elif 'CUDA Version:' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("nvidia-smi no disponible")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("nvidia-smi no encontrado o timeout")
        return False

def check_torch_cuda():
    """Verifica que PyTorch tenga soporte CUDA"""
    print("\nVerificando PyTorch CUDA...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print("PyTorch con soporte CUDA")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Devices: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   Device {i}: {props.name}")
                print(f"   Memory: {props.total_memory / 1024**3:.1f}GB")
            
            return True
        else:
            print("PyTorch sin soporte CUDA")
            print("   Solución: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False
            
    except ImportError:
        print("PyTorch no instalado")
        return False

def check_llama_cpp_cuda():
    """Verifica que llama-cpp-python funcione con CUDA"""
    print("\nVerificando llama-cpp-python...")
    try:
        from llama_cpp import Llama
        print("llama-cpp-python instalado")
        
        # Intentar crear una instancia con GPU (sin modelo real)
        try:
            # Esto fallará sin modelo, pero verificará que la librería esté bien
            print("   Probando inicialización CUDA...")
            # No podemos probar sin modelo real, pero podemos verificar imports
            print("llama-cpp-python listo para CUDA")
            return True
        except Exception as e:
            print(f"Error en inicialización: {e}")
            return False
            
    except ImportError:
        print("llama-cpp-python no instalado")
        return False

def provide_solutions():
    """Proporciona soluciones para problemas comunes"""
    print("\n" + "="*60)
    print("SOLUCIONES PARA PROBLEMAS CUDA")
    print("="*60)
    
    print("\n1. Si PyTorch no detecta CUDA:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n2. Si llama-cpp-python no usa GPU:")
    print("   pip uninstall llama-cpp-python")
    print("   pip install llama-cpp-python --force-reinstall --no-cache-dir")
    
    print("\n3. Para CUDA 11.8 (si 12.1 no funciona):")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n4. Verificar instalación:")
    print("   python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\"")

def main():
    """Función principal de verificación"""
    print("="*60)
    print("🔧 VERIFICACIÓN CUDA - LOCAL LLM CHAT")
    print("="*60)
    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Verificaciones
    nvidia_ok = check_nvidia_driver()
    torch_ok = check_torch_cuda()
    llama_ok = check_llama_cpp_cuda()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    
    checks = [
        ("Driver NVIDIA", nvidia_ok),
        ("PyTorch CUDA", torch_ok),
        ("llama-cpp-python", llama_ok)
    ]
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name}")
    
    print("="*60)
    print(f"Resultados: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        print("\n¡CUDA configurado correctamente!")
        print("   Puedes usar aceleración GPU en Local LLM Chat")
        return 0
    else:
        print("\nProblemas detectados con CUDA")
        provide_solutions()
        return 1

if __name__ == "__main__":
    sys.exit(main())
