# Local LLM Chat - Universal Local LLM Interface

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GGUF](https://img.shields.io/badge/format-GGUF-green.svg)](https://github.com/ggerganov/ggml)
[![Transformers](https://img.shields.io/badge/backend-transformers-orange.svg)](https://huggingface.co/docs/transformers)

Una interfaz universal para ejecutar modelos de lenguaje localmente. Soporta **modelos GGUF** (vía llama.cpp) y **Transformers** (en el futuro) de Hugging Face. Diseñado con adaptación automática de system prompt y detección inteligente de modelos.

## Características Principales

- **Soporte Universal de GGUF**: Compatible con Llama, Mistral, Gemma, Phi, Qwen y más
- **Detección Automática**: Reconoce el tipo de modelo y adapta el formato de chat automáticamente
- **System Prompts Inteligentes**: Maneja modelos con y sin soporte nativo de system prompt
- **Consciente del Hardware**: Recomienda modelos óptimos según RAM/VRAM disponible
- **RAG (Retrieval-Augmented Generation)**: Dos backends para Q&A sobre documentos
  - SimpleRAG: ChromaDB, rápido, optimizado para CPU
  - RAG-Anything: Knowledge graph, complejo, para GPU
- **Configuración Centralizada**: Sistema híbrido (código, JSON, env vars)
- **Persistencia de Documentos**: RAG recuerda documentos entre sesiones
- **Gestión de Sesiones**: Registro automático de conversaciones con métricas completas
- **Presets Configurables**: System prompts preconfigurados para diferentes casos de uso
- **Cambio Dinámico de Modelos**: Cambia entre modelos sin perder configuración
- **Aceleración GPU**: Soporte automático para Metal (macOS) y CUDA (Linux/Windows)

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

- Python 3.8 o superior
- 4GB RAM mínimo (8GB+ recomendado)
- 10GB espacio en disco para modelos
- Opcional: GPU con soporte CUDA o Metal

## Instalación

### Instalación Estándar

```bash
# Clonar el repositorio
git clone https://github.com/RGiskard7/local-llm-chat.git
cd local-llm-chat

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -e .
```

### Desde PyPI

```bash
pip install local-llm-chat
```

### Verificar Instalación

```bash
python verify_installation.py
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

```python
from local_llm_chat import UniversalChatClient

# Inicializar cliente
client = UniversalChatClient(
    model_path="models/llama-3.1-8b-instruct.gguf",
    system_prompt="Eres un asistente experto en Python."
)

# Generar respuesta
response = client.infer("¿Qué es un decorador?")
print(response)

# Guardar sesión
client.save_log()
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

## Modelos Soportados

### Soporte Nativo Completo

| Familia | Versiones | System Prompt | Formato |
|---------|-----------|---------------|---------|
| Llama | 2, 3, 3.1 | Nativo | llama-2/3 |
| Mistral | 7B, Mixtral | Nativo | mistral |
| OpenChat | 3.5+ | Nativo | openchat |

### Soporte con Workaround

| Familia | Versiones | System Prompt | Formato |
|---------|-----------|---------------|---------|
| Gemma | 1, 2, 3 | Workaround | gemma |
| Phi | 3, 3.5 | Workaround | phi |
| Qwen | 2, 2.5 | ChatML | chatml |

### Otros Modelos

- **Dolphin/Nous-Hermes**: Soporte ChatML
- **Yi**: Soporte ChatML
- **Modelos desconocidos**: Auto-detección con fallback

## Configuración

### System Prompts Personalizados

Editar `src/chat_ia/prompts.py`:

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
│       ├── simple.py         # SimpleRAG (ChromaDB)
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
pytest --cov=src/chat_ia

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
ModuleNotFoundError: No module named 'chat_ia'
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

### Error: GPU No Detectada

**Soluciones para CUDA:**

```bash
# Verificar instalación CUDA
nvidia-smi

# Reinstalar llama-cpp-python con soporte CUDA
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

- Versión actual: 1.0.0
- Estado: Estable
- Python: 3.8+
- Última actualización: Octubre 2025

-----

<p align="center">
  <small>Desarrollado por <b>Edu Díaz</b> (<b>RGiskard7</b>) ❤️</small>
</p>
