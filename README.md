# Local LLM Chat - Universal Multi-Backend LLM Interface

[![Python 3.8-3.13](https://img.shields.io/badge/python-3.8--3.13-blue.svg)](https://www.python.org/downloads/) [![RAG: 3.11-3.12](https://img.shields.io/badge/RAG-3.11--3.12-orange.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GGUF](https://img.shields.io/badge/format-GGUF-green.svg)](https://github.com/ggerganov/ggml)
[![Transformers](https://img.shields.io/badge/backend-transformers-orange.svg)](https://huggingface.co/docs/transformers)
[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/RGiskard7/local-llm-chat)

Una interfaz universal para ejecutar modelos de lenguaje localmente con **múltiples backends intercambiables**. Soporta **modelos GGUF** (vía llama.cpp) y **Transformers** (Hugging Face). Diseñado con adaptación automática de system prompt, detección inteligente de modelos y arquitectura modular.

## Características Principales

### 🚀 **NUEVO v2.0**: Sistema Multi-Backend
- **GGUF Backend**: Modelos cuantizados vía llama-cpp-python (original)
- **Transformers Backend**: Modelos Hugging Face (local o remoto) - **NUEVO**
- **Intercambiabilidad Total**: Cambia de backend sin modificar código
- **Interfaz Común**: Misma API para ambos backends

### 💡 Características Core
- **Soporte Universal**: GGUF y Transformers con cualquier arquitectura
- **Detección Automática**: Reconoce tipo de modelo y backend automáticamente
- **System Prompts Inteligentes**: Adaptación automática según capacidades del modelo
- **Consciente del Hardware**: Recomendaciones según RAM/VRAM disponible
- **RAG (Retrieval-Augmented Generation)**: Compatible con ambos backends
  - SimpleRAG: ChromaDB, rápido, optimizado para CPU
  - RAG-Anything: Knowledge graph, complejo, para GPU
- **Configuración Centralizada**: Sistema híbrido (código, JSON, env vars)
- **Persistencia de Documentos**: RAG recuerda documentos entre sesiones
- **Gestión de Sesiones**: Registro automático con métricas completas
- **Presets Configurables**: System prompts preconfigurados
- **Cambio Dinámico**: Cambia modelos y backends durante la sesión
- **Aceleración GPU**: CUDA (NVIDIA) y Metal (Apple Silicon)
- **Cuantización**: Soporte 8-bit/4-bit para Transformers (opcional)

## Tabla de Contenidos

- [Instalación](#instalación)
- [Inicio Rápido](#inicio-rápido)
- [Uso](#uso)
- [Comandos CLI](#comandos-cli)
- [Uso como Biblioteca](#uso-como-biblioteca)
- [Modelos Soportados](#modelos-soportados)
- [Configuración](#configuración)
- [Desarrollo](#desarrollo)
- [Solución de Problemas](#solución-de-problemas)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

## Requisitos

- **Python 3.8 - 3.12** (Core + GGUF + Transformers)
- **Python 3.11 - 3.12** (RAG - incompatible con 3.13)
- 4GB RAM mínimo (8GB+ recomendado)
- 10GB espacio en disco para modelos
- Opcional: GPU con soporte CUDA o Metal

### ⚠️ Nota sobre Python 3.13

El **core del proyecto** (GGUF + Transformers) funciona perfectamente con Python 3.13. Sin embargo, **las funcionalidades RAG** requieren Python 3.11 o 3.12 debido a dependencias incompatibles (`lightrag-hku` → `future` antiguo).

### Requisitos GPU (CUDA)

Para usar aceleración GPU en Windows/Linux con NVIDIA:

1. **CUDA Toolkit** instalado (versión 11.8 o 12.1)
2. **PyTorch con soporte CUDA** (ver instalación abajo)
3. **Drivers NVIDIA** actualizados

**⚠️ Importante**: Por defecto, PyTorch se instala sin soporte CUDA. Debes instalarlo explícitamente.

## Instalación

### Instalación Estándar (GGUF Backend)

```bash
# 1. Clonar repositorio
git clone https://github.com/RGiskard7/local-llm-chat.git
cd local-llm-chat

# 2. Crear entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Instalar dependencias básicas (solo GGUF)
pip install -e .
```

**Eso es todo para GGUF.** Solo necesitas 3 dependencias:
- `llama-cpp-python` - Modelos GGUF
- `huggingface-hub` - Descargar modelos
- `psutil` - Detectar hardware

### Instalación con Transformers Backend

```bash
# Instalar con soporte Transformers (incluye transformers + accelerate)
pip install -e ".[transformers]"

# O con cuantización 8-bit/4-bit (incluye transformers + accelerate + bitsandbytes)
pip install -e ".[quantization]"

# O todo (Transformers + RAG + cuantización)
pip install -e ".[all]"
```

**Nota**: `accelerate` se instala automáticamente con `[transformers]` o `[quantization]`. Es necesario para:
- Gestión eficiente de memoria
- Balanceo automático entre dispositivos (GPU/CPU)
- Soporte para modelos grandes

Si instalas solo las dependencias básicas (`pip install -e .`), los modelos Transformers funcionarán pero con selección manual de dispositivo (sin `device_map="auto"`).

### Instalación con RAG (Python 3.11/3.12 solamente)

⚠️ **Las funcionalidades RAG requieren Python 3.11 o 3.12** (incompatibles con Python 3.13):

```bash
# Verificar versión de Python
python --version  # Debe ser 3.11.x o 3.12.x

# Instalar dependencias RAG por separado
pip install -r requirements-rag.txt

# O usar pyproject.toml extras
pip install -e ".[rag]"
```

**Si tienes Python 3.13**: El core del proyecto funciona perfectamente, pero RAG no estará disponible hasta que `lightrag-hku` se actualice.

### Instalación con Soporte CUDA (Windows/Linux)

Para usar aceleración GPU con NVIDIA:

```bash
# 1. Instalar PyTorch con CUDA primero
pip uninstall torch torchvision torchaudio  # Si ya está instalado
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Instalar dependencias del proyecto
pip install -e ".[transformers]"

# 3. Verificar CUDA
python verify_cuda.py
```

### (Opcional) Funcionalidad RAG

Si quieres Q&A sobre documentos:

```bash
pip install chromadb sentence-transformers pypdf
```

Esto permite:
- Cargar PDFs y TXT con `/load archivo.pdf`
- Hacer preguntas sobre el contenido
- Búsqueda semántica en documentos

### Verificar CUDA (opcional)

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Inicio Rápido

### 1. Primera Ejecución

```bash
python main.py
```

La aplicación mostrará recomendaciones inteligentes basadas en tu hardware:

```
RECOMMENDED MODELS (based on your hardware):
  1. bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
     Size: ~9.0GB | Type: llama-3
     Downloads: 2,547,891
     Use: /download 1
```

### 2. Descargar y Chatear

```bash
# En el prompt, descargar modelo recomendado
> /download 1

# Esperar descarga
[DOWNLOAD] Downloading model...
[READY] Model loaded successfully

# Comenzar conversación
> Hola, ¿cómo estás?
[LLAMA-3] Hola, estoy funcionando correctamente. ¿En qué puedo ayudarte?
```

### 3. Usar System Prompts

```bash
# Cargar preset de programación
> /preset coding

# Hacer una pregunta técnica
> ¿Cómo implemento un decorador en Python?
```

## Uso

### Modo CLI

```bash
# Ejecutar directamente
python main.py

# Como módulo Python
python -m local_llm_chat

# Comando instalado
local-llm-chat
# o el alias corto:
llm-chat
```

### Modo Biblioteca

#### Backend GGUF (modelos locales .gguf)

```python
from local_llm_chat import UniversalChatClient

# Inicializar con GGUF usando model_path (recomendado)
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.1-8b-instruct.gguf",
    system_prompt="Eres un asistente experto en Python."
)

# Generar respuesta
response = client.infer("¿Qué es un decorador?")
print(response)
```

#### Backend Transformers (modelos Hugging Face)

```python
from local_llm_chat import UniversalChatClient

# Modelo remoto desde HuggingFace Hub
# Puedes usar model_name_or_path (convención HF) o model_path
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m",  # Recomendado para HF
    system_prompt="Eres un asistente útil."
)

# También funciona con model_path (son aliases)
client = UniversalChatClient(
    backend="transformers",
    model_path="bigscience/bloom-560m",  # También válido
    system_prompt="Eres un asistente útil."
)

# Modelo local
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="/path/to/local/model",
    device="cuda",
    torch_dtype="float16"
)

# Con cuantización 8-bit (requiere bitsandbytes)
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    load_in_8bit=True
)

response = client.infer("Explícame la IA")
print(response)
```

**Nota**: `model_path` y `model_name_or_path` son **intercambiables** - ambos funcionan con ambos backends. Usa el que prefieras o el que sea más natural para tu caso.

#### Cambio Dinámico de Backend

```python
# Iniciar con GGUF
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b.gguf"
)

# Cambiar a Transformers (usa model_name_or_path o model_path)
client.change_model(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"  # O model_path="..."
)

# Volver a GGUF
client.change_model(
    backend="gguf",
    model_path="models/mistral-7b.gguf"
)
```

## Comandos CLI

### Conversación

| Comando | Descripción |
|---------|-------------|
| `/exit` | Salir y guardar registro |
| `/save` | Guardar conversación ahora |
| `/history` | Mostrar historial completo |
| `/clear` | Limpiar historial actual |
| `/stats` | Mostrar estadísticas de sesión |
| `/help` | Mostrar ayuda de comandos |

### Gestión de Modelos

| Comando | Descripción |
|---------|-------------|
| `/models` | Listar modelos disponibles y recomendaciones |
| `/download <num>` | Descargar modelo recomendado por número |
| `/changemodel <path>` | Cambiar a modelo diferente |

### System Prompts

| Comando | Descripción |
|---------|-------------|
| `/system <texto>` | Establecer system prompt personalizado |
| `/showsystem` | Mostrar prompt actual |
| `/clearsystem` | Eliminar system prompt |
| `/preset <name>` | Cargar preset preconfigurado |
| `/presets` | Listar presets disponibles |

### RAG (Document Q&A)

| Comando | Descripción |
|---------|-------------|
| `/load <file>` | Cargar documento en RAG |
| `/unload <file>` | Eliminar documento del RAG |
| `/list` | Listar documentos cargados |
| `/clear` | Limpiar todos los documentos |
| `/rag on` | Activar modo RAG |
| `/rag off` | Desactivar modo RAG |
| `/status` | Estado del sistema RAG |

**Workflow RAG:**
1. `/load documento.pdf` - Carga documento
2. `/rag on` - Activa búsqueda en documentos
3. Hacer preguntas - El sistema busca contexto relevante
4. `/rag off` - Desactiva RAG (chat libre)
5. Documentos persisten entre sesiones

## Uso como Biblioteca

### Ejemplo Básico

```python
from local_llm_chat import UniversalChatClient

# Crear cliente con modelo local
client = UniversalChatClient(
    model_path="./models/llama-3.1-8b-instruct.gguf"
)

# Conversación simple
respuesta = client.infer("Explícame qué es Python")
print(respuesta)
```

### Ejemplo con System Prompt

```python
from local_llm_chat import UniversalChatClient

# Cliente con comportamiento específico
client = UniversalChatClient(
    model_path="./models/mistral-7b-instruct.gguf",
    system_prompt="Eres un experto en desarrollo backend con Python y FastAPI."
)

# Hacer preguntas específicas
respuesta = client.infer("¿Cómo implemento autenticación JWT en FastAPI?")
print(respuesta)
```

### Ejemplo Avanzado

```python
from local_llm_chat import UniversalChatClient, get_hardware_info

# Verificar hardware disponible
hw = get_hardware_info()
print(f"RAM disponible: {hw['ram_available_gb']}GB")

# Crear cliente
client = UniversalChatClient(
    model_path="./models/llama-3.1-8b-instruct.gguf",
    n_gpu_layers=-1  # Usar todas las capas GPU disponibles
)

# Cargar preset
client.load_preset("coding")

# Conversación multi-turno
preguntas = [
    "¿Qué es un closure en Python?",
    "Dame un ejemplo práctico",
    "¿Cuándo debería usarlo?"
]

for pregunta in preguntas:
    respuesta = client.infer(pregunta)
    print(f"P: {pregunta}")
    print(f"R: {respuesta}\n")

# Guardar toda la conversación
client.save_log()
```

## Backends Soportados

### GGUF Backend (llama-cpp-python)

Modelos cuantizados locales en formato .gguf:

| Familia | Versiones | System Prompt | Formato |
|---------|-----------|---------------|---------|
| Llama | 2, 3, 3.1 | Nativo | llama-2/3 |
| Mistral | 7B, Mixtral | Nativo | mistral |
| OpenChat | 3.5+ | Nativo | openchat |
| Gemma | 1, 2, 3 | Workaround | gemma |
| Phi | 3, 3.5 | Workaround | phi |
| Qwen | 2, 2.5 | ChatML | chatml |
| Dolphin/Nous-Hermes | - | ChatML | chatml |
| Yi | - | ChatML | chatml |

### Transformers Backend (Hugging Face)

Cualquier modelo de Hugging Face compatible con `AutoModelForCausalLM`:

**Familias populares**:
- **GPT**: GPT-2, GPT-Neo, GPT-J, GPT-NeoX, GPT-4 (community)
- **Llama**: Llama-2, Llama-3, Vicuna, Alpaca, WizardLM
- **Mistral**: Mistral-7B, Mixtral-8x7B, Zephyr
- **Bloom**: BLOOM-560m, BLOOM-1b7, BLOOM-7b1
- **Falcon**: Falcon-7B, Falcon-40B, Falcon-180B
- **Phi**: Phi-1.5, Phi-2, Phi-3
- **Gemma**: Gemma-2B, Gemma-7B
- **Qwen**: Qwen-7B, Qwen-14B, Qwen-72B
- **MPT**: MPT-7B, MPT-30B
- **StableLM**: StableLM-7B, StableLM-Alpha
- Y muchos más...

**Ejemplos de uso**:
```python
# Modelos pequeños (< 1GB)
"bigscience/bloom-560m"
"EleutherAI/gpt-neo-125M"
"microsoft/phi-2"

# Modelos medianos (1-10GB)
"bigscience/bloom-1b7"
"EleutherAI/gpt-j-6B"
"mistralai/Mistral-7B-v0.1"

# Modelos grandes (> 10GB)
"meta-llama/Llama-2-7b-hf"
"tiiuae/falcon-7b"
```

### Comparación de Backends

| Aspecto | GGUF | Transformers |
|---------|------|--------------|
| **Formato** | .gguf cuantizado | PyTorch/SafeTensors |
| **Tamaño típico** | 2-8GB (cuantizado) | 10-30GB (full precision) |
| **Velocidad CPU** | ⚡⚡⚡ Muy rápido | 🟡 Medio |
| **Velocidad GPU** | ⚡⚡ Rápido | ⚡⚡⚡ Muy rápido |
| **RAM necesaria** | ✅ Baja (2-8GB) | ❌ Alta (8-32GB) |
| **Fuente** | Solo local | Local o HF Hub |
| **Cuantización** | Nativa (Q4, Q5, Q8) | Requiere bitsandbytes |
| **Instalación** | Incluida | Opcional |

**Recomendaciones**:
- **Usa GGUF** para máxima velocidad en CPU y bajo uso de RAM
- **Usa Transformers** para acceso a cualquier modelo HF o para experimentación

## Configuración

### Sistema de Configuración Centralizada

El proyecto usa un sistema de configuración híbrido con tres secciones:

1. **Model**: Parámetros de carga del modelo (n_ctx, n_gpu_layers, verbose)
2. **LLM**: Parámetros de inferencia (max_tokens, temperature, top_p, etc.)
3. **RAG**: Parámetros de documentos (chunk_size, top_k, etc.)

Ver [CONFIG.md](CONFIG.md) para documentación completa.

### System Prompts Personalizados

Editar `src/local_llm_chat/prompts.py`:

```python
PROMPTS = {
    "coding": """Eres un programador experto especializado en Python.
    Proporcionas código limpio, bien documentado y siguiendo PEP 8.""",

    "creative": """Eres un escritor creativo. Generas contenido original,
    descriptivo y envolvente.""",

    "tutor": """Eres un tutor paciente y didáctico. Explicas conceptos
    complejos de forma simple y con ejemplos prácticos.""",
}
```

### Detección de Hardware

El sistema detecta automáticamente:

- **RAM Total y Disponible**: Para recomendar tamaño de modelo apropiado
- **GPU/VRAM**: CUDA (NVIDIA) o Metal (Apple Silicon)
- **Capacidades del Sistema**: Número de cores, arquitectura

### Aceleración GPU

#### macOS (Apple Silicon)

```python
# Automático - usa Metal acceleration
client = UniversalChatClient(model_path="...")
```

#### Linux/Windows (NVIDIA)

```python
# Automático - usa CUDA si está disponible
client = UniversalChatClient(
    model_path="...",
    n_gpu_layers=-1  # Todas las capas en GPU
)
```

#### CPU Only

```python
client = UniversalChatClient(
    model_path="...",
    n_gpu_layers=0  # Solo CPU
)
```

## Estructura del Proyecto

```
local-llm-chat/
├── src/local_llm_chat/       # Código fuente principal
│   ├── client.py             # Clase UniversalChatClient
│   ├── cli.py                # Interfaz de línea de comandos
│   ├── model_config.py       # Detección y configuración de modelos
│   ├── prompts.py            # System prompts preconfigurados
│   ├── utils.py              # Funciones auxiliares
│   ├── config.py             # Sistema de configuración
│   ├── config.json           # Configuración por defecto
│   └── rag/                  # Módulo RAG
│       ├── base.py           # Interfaz RAGBackend
│       ├── simple_rag_backend.py # SimpleRAG (ChromaDB)
│       ├── raganything_backend.py  # RAG-Anything
│       └── manager.py        # RAGManager
├── tests/                    # Suite de pruebas
├── models/                   # Modelos GGUF (gitignored)
├── chat_logs/                # Registros de sesiones (gitignored)
├── simple_rag_data/          # Datos SimpleRAG (gitignored)
├── rag_data/                 # Datos RAG-Anything (gitignored)
├── main.py                   # Punto de entrada principal
├── pyproject.toml            # Configuración del paquete
├── requirements.txt          # Dependencias
├── CONFIG.md                 # Guía de configuración
└── changelog.md              # Historial de cambios
```

Ver [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) para más detalles.

## Desarrollo

### Configuración del Entorno

```bash
# Instalar en modo desarrollo
pip install -e ".[dev]"

# Instalar dependencias de desarrollo
pip install pytest black mypy flake8
```

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Con cobertura
pytest --cov=src/local_llm_chat

# Tests específicos
pytest tests/test_model_config.py
```

### Formateo y Linting

```bash
# Formatear código
black src/ tests/

# Verificar estilo
flake8 src/ tests/

# Verificar tipos
mypy src/
```

### Estructura de Tests

```python
# tests/test_client.py
import pytest
from local_llm_chat import UniversalChatClient

def test_client_initialization():
    """Prueba inicialización básica del cliente"""
    client = UniversalChatClient(
        model_path="tests/fixtures/tiny_model.gguf"
    )
    assert client is not None
    assert client.model_type is not None
```

## Registros de Sesión

Las conversaciones se guardan automáticamente en `./chat_logs/` con formato JSON:

```json
{
  "session_id": "20250118_143052",
  "model_type": "llama-3",
  "chat_format": "llama-3",
  "supports_native_system": true,
  "preset_name": "coding",
  "session_start": "2025-01-18T14:30:52",
  "session_end": "2025-01-18T15:15:30",
  "total_messages": 12,
  "conversation": [
    {
      "timestamp": "2025-01-18T14:31:15",
      "user": "¿Qué es un decorador en Python?",
      "assistant": "Un decorador en Python es...",
      "metrics": {
        "elapsed_seconds": 2.3,
        "prompt_tokens": 45,
        "completion_tokens": 120,
        "total_tokens": 165
      }
    }
  ]
}
```

## Solución de Problemas

### Error: Modelo No Carga

```bash
[ERROR] Failed to load model: unknown model architecture
```

**Soluciones:**

1. Actualizar llama-cpp-python:
   ```bash
   pip install --upgrade llama-cpp-python
   ```

2. Verificar compatibilidad del modelo:
   ```bash
   python -c "from local_llm_chat import detect_model_type; print(detect_model_type('modelo.gguf'))"
   ```

3. Descargar modelo compatible:
   ```bash
   python main.py
   # Usar /models para ver recomendaciones
   ```

### Error: Sin Memoria

```bash
[ERROR] Insufficient RAM/VRAM (try a smaller model)
```

**Soluciones:**

1. Ver modelos compatibles con tu hardware:
   ```bash
   > /models
   ```

2. Descargar cuantización más pequeña (Q4 en lugar de Q8)

3. Reducir contexto:
   ```python
   client = UniversalChatClient(
       model_path="...",
       n_ctx=2048  # Reducir de 8192 a 2048
   )
   ```

### Error: Importación de Módulo

```bash
ModuleNotFoundError: No module named 'local_llm_chat'
```

**Soluciones:**

1. Instalar en modo desarrollo:
   ```bash
   pip install -e .
   ```

2. Verificar instalación:
   ```bash
   python verify_installation.py
   ```

3. Usar ejecutable directo:
   ```bash
   python main.py
   ```

### Error: accelerate no instalado (Transformers)

```bash
ValueError: Using a `device_map` requires `accelerate`. 
You can install it with `pip install accelerate`
```

**Causa**: Estás intentando usar modelos Transformers sin tener `accelerate` instalado.

**Soluciones:**

1. **Recomendado**: Instalar con soporte completo Transformers:
   ```bash
   pip install -e ".[transformers]"
   ```

2. **Alternativa**: Instalar solo accelerate:
   ```bash
   pip install accelerate
   ```

3. **Sin accelerate**: El sistema tiene un fallback automático que funciona sin `accelerate`, pero con gestión de memoria menos eficiente. Si ves este error, el fallback debería activarse automáticamente.

**Nota**: `accelerate` es necesario para:
- Gestión eficiente de memoria
- Balanceo automático entre GPU/CPU
- Modelos grandes

### Error: GPU No Detectada

**Problema más común**: PyTorch instalado sin soporte CUDA.

**Solución para CUDA (Windows/Linux):**

```bash
# 1. Verificar instalación CUDA
nvidia-smi

# 2. Desinstalar PyTorch CPU-only
pip uninstall torch torchvision torchaudio

# 3. Instalar PyTorch con CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Verificar instalación
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 5. Si es necesario, reinstalar llama-cpp-python
pip uninstall llama-cpp-python
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Soluciones para Metal (macOS):**

```bash
# Verificar que estás en Apple Silicon
uname -m  # Debe mostrar 'arm64'

# Reinstalar con soporte Metal
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Verificación completa:**

```bash
# Verificar hardware detectado
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
```

## Contribuir

Las contribuciones son bienvenidas. Por favor sigue estos pasos:

### Proceso

1. **Fork** el repositorio
2. **Crea** una rama (`git checkout -b feature/nueva-caracteristica`)
3. **Commit** cambios (`git commit -m 'Agregar nueva característica'`)
4. **Push** a la rama (`git push origin feature/nueva-caracteristica`)
5. **Abre** un Pull Request

### Guías

- Seguir PEP 8 para estilo de código Python
- Agregar tests para nuevas funcionalidades
- Actualizar documentación según corresponda
- Mantener compatibilidad con Python 3.8+

### Áreas de Contribución

- Soporte para nuevos modelos
- Mejoras en detección de hardware
- Optimización de rendimiento
- Corrección de bugs
- Documentación y ejemplos
- Tests adicionales

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 Local LLM Chat Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## Agradecimientos

- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)**: Binding de Python para llama.cpp
- **[Hugging Face](https://huggingface.co)**: Hosting de modelos GGUF
- **Comunidad Open Source**: Por compartir modelos y conocimiento

## Roadmap

### Versión 1.1

- [x] Soporte RAG (Retrieval-Augmented Generation)
- [x] Sistema de configuración centralizada
- [x] Persistencia de documentos RAG entre sesiones
- [ ] Interfaz web con Gradio
- [ ] Exportar conversaciones a Markdown/PDF

### Versión 1.2

- [ ] API REST con FastAPI
- [ ] Sistema de plugins
- [ ] Integración con Langchain
- [ ] Soporte para conversaciones multi-modelo

### Versión 2.0

- [ ] Fine-tuning de modelos locales
- [ ] Suite de benchmarking integrada
- [ ] Soporte multimodal (imágenes, audio)

## Contacto y Soporte

- **Issues**: [GitHub Issues](https://github.com/RGiskard7/local-llm-chat/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/RGiskard7/local-llm-chat/discussions)

## Estado del Proyecto

- Versión actual: 2.0.0 🎉
- Estado: Estable
- Python: 3.8+
- Backends: GGUF + Transformers
- Última actualización: Noviembre 2025

## Novedades v2.0

### 🚀 Sistema Multi-Backend
- ✅ Arquitectura modular con backends intercambiables
- ✅ Backend Transformers (Hugging Face) totalmente funcional
- ✅ Interfaz común para ambos backends
- ✅ System prompts adaptativos universales
- ✅ RAG compatible con ambos backends
- ✅ Cambio dinámico de backend durante la sesión
- ✅ Detección automática de tipo de backend
- ✅ Soporte cuantización 8-bit/4-bit para Transformers

### 📚 Documentación
- ✅ [BACKENDS_ARCHITECTURE.md](doc/BACKENDS_ARCHITECTURE.md) - Guía completa de backends
- ✅ README actualizado con ejemplos de uso
- ✅ Instalación modular con dependencias opcionales

-----

<p align="center">
  <small>Desarrollado por <b>Edu Díaz</b> (<b>RGiskard7</b>) ❤️</small>
</p>
