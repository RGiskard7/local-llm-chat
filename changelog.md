# Changelog - Local LLM Chat

## 📅 2025-10-20 — Mejoras CUDA y Documentación

**Archivos modificados:**
- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `QUICKSTART.md`
- `verify_cuda.py` (nuevo)

**Resumen:**
Mejorada la documentación y configuración para soporte CUDA en Windows/Linux. Agregado script de verificación CUDA y instrucciones detalladas para resolver problemas comunes de GPU.

**Cambios principales:**
- ✅ Agregadas instrucciones específicas para instalar PyTorch con CUDA
- ✅ Creado script `verify_cuda.py` para diagnóstico de problemas GPU
- ✅ Actualizado `requirements.txt` con comentarios sobre CUDA
- ✅ Agregadas dependencias CUDA opcionales en `pyproject.toml`
- ✅ Mejorada sección de troubleshooting en README
- ✅ Actualizada guía de inicio rápido con pasos CUDA

**Motivación:**
Los usuarios con GPU NVIDIA (como RTX 4070) experimentaban problemas porque PyTorch se instalaba sin soporte CUDA por defecto. Esto causaba que el sistema mostrara "GPU: Not available (CPU only)" incluso con hardware compatible.

**Solución implementada:**
1. Documentación clara sobre instalación de PyTorch con CUDA
2. Script automatizado para verificar configuración GPU
3. Instrucciones paso a paso para resolver problemas comunes
4. Dependencias opcionales para diferentes versiones de CUDA

**Próximos pasos:**
- Considerar automatizar la detección e instalación de CUDA
- Agregar soporte para más versiones de CUDA
- Mejorar detección automática de hardware GPU
