# Guía de Inicio Rápido - Local LLM Chat v2.0

Instrucciones para poner en marcha Local LLM Chat en pocos minutos.

## 🆕 Novedades v2.0

- ✅ **Múltiples backends**: GGUF (llama.cpp) + Transformers (Hugging Face)
- ✅ **Python 3.13**: Core funciona perfectamente
- ✅ **RAG**: Disponible en Python 3.11/3.12
- ✅ **Intercambiabilidad**: Cambia entre modelos GGUF y Transformers sin cambiar código

## Requisitos

- **Python 3.8 - 3.13** (Core + GGUF + Transformers)
- **Python 3.11 - 3.12** (Para usar RAG)
- 4GB RAM mínimo (8GB+ recomendado)

## Instalación Básica (GGUF)

```bash
# 1. Clonar o navegar al proyecto
cd local-llm-chat

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# 3. Instalar solo GGUF (funciona en Python 3.13)
pip install -e .

# 4. ✅ Listo para modelos GGUF!
```

## Instalación Completa (GGUF + Transformers)

```bash
# Después de los pasos 1-2 anteriores:

# 3. Instalar con backend Transformers (incluye transformers + accelerate)
pip install -e ".[transformers]"

# O con TODO (Transformers + cuantización + accelerate + bitsandbytes)
pip install -e ".[quantization]"

# 4. ✅ Listo para GGUF y Transformers!
```

**¿Qué incluye cada instalación?**
- `[transformers]`: transformers + accelerate (gestión de memoria y balanceo de dispositivos)
- `[quantization]`: transformers + accelerate + bitsandbytes (cuantización 8-bit/4-bit)

**Nota sobre `accelerate`**: Se instala automáticamente con `[transformers]`. Es necesario para usar `device_map="auto"` y gestión eficiente de memoria. Sin él, los modelos Transformers funcionan pero con selección manual de dispositivo.

## Instalación con RAG (Python 3.11/3.12 solamente)

⚠️ **Importante**: RAG requiere Python 3.11 o 3.12 (no funciona en 3.13)

```bash
# Verificar versión
python --version  # Debe ser 3.11.x o 3.12.x

# Instalar dependencias RAG
pip install -r requirements-rag.txt

# ✅ Listo con RAG!
```

### Para GPU (CUDA) - Windows/Linux

```bash
# Instalar PyTorch con CUDA primero
pip uninstall torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verificar CUDA
python verify_cuda.py
```

## Primera Ejecución

### Opción A: Modelo GGUF Local

```bash
# Ejecutar la aplicación
python main.py

# Seleccionar modelo GGUF de la lista
# Comenzar a chatear
```

### Opción B: Modelo Transformers (Hugging Face)

```bash
# Ejecutar con modelo de HuggingFace
python main.py --backend transformers --model "bigscience/bloom-560m"

# O especificar en la CLI durante ejecución
python main.py
# Luego: /changemodel --backend transformers --model "microsoft/DialoGPT-small"
```

### Opción C: Descargar un modelo recomendado

```bash
# Ejecutar la aplicación
python main.py

# La aplicación mostrará recomendaciones basadas en tu RAM (GGUF y Transformers)
# Opción 1: Seleccionar un número para descargar
# Opción 2: Descargar directamente por ID de HuggingFace
> /download meta-llama/Llama-3.1-8B-Instruct-GGUF
> /download bigscience/bloom-560m

# Esperar la descarga (puede tomar varios minutos)
# Comenzar a chatear
```

## Uso Básico

```
> Hola

[LLAMA-3] Hola, ¿cómo puedo ayudarte hoy?

> /help              # Mostrar todos los comandos
> /stats             # Mostrar estadísticas de la sesión
> /history           # Mostrar historial de conversación
> /exit              # Guardar y salir
```

## Comandos Comunes

### System Prompts

```bash
/preset coding      # Cargar preset de asistente de programación
/preset creative    # Cargar preset de escritura creativa
/system Eres un experto en Python    # Prompt personalizado
/showsystem         # Ver prompt actual
```

### Gestión de Modelos

```bash
/models                           # Listar modelos locales y recomendaciones (GGUF y Transformers)
/download 1                       # Descargar modelo recomendado por número
/download meta-llama/Llama-3.1-8B-Instruct-GGUF  # Descargar por ID de HuggingFace
/download bigscience/bloom-560m   # Descargar modelo Transformers directamente
/changemodel models/model.gguf    # Cambiar a GGUF local

# Cambiar a Transformers
/changemodel --backend transformers --model "meta-llama/Llama-2-7b-chat-hf"
/changemodel --backend transformers --model "bigscience/bloom-560m"
```

### Gestión de Sesión

```bash
/save               # Guardar conversación ahora
/clear              # Limpiar historial
/stats              # Mostrar estadísticas
/exit               # Guardar y salir
```

## Uso como Biblioteca

### Ejemplo 1: Modelo GGUF Local

```python
from local_llm_chat import UniversalChatClient, Config

# Cargar configuración
config = Config()

# Backend GGUF (por defecto)
client = UniversalChatClient(
    backend="gguf",  # o simplemente omitir (es el default)
    model_path="models/llama-3.1-8b-instruct.gguf",
    system_prompt="Eres un asistente útil.",
    n_ctx=config.model.n_ctx,
    n_gpu_layers=config.model.n_gpu_layers,
    verbose=config.model.verbose,
    llm_config=config.llm
)

# Generar respuesta
response = client.infer("¿Qué es Python?")
print(response)

# Guardar sesión
client.save_log()
```

### Ejemplo 2: Modelo Transformers (Hugging Face)

```python
from local_llm_chat import UniversalChatClient

# Backend Transformers
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="microsoft/DialoGPT-small",  # o "model_path"
    system_prompt="Eres un asistente experto en Python.",
    device="cuda"  # o "cpu" o "mps" (Mac)
)

# Generar respuesta
response = client.infer("Explica las list comprehensions")
print(response)
```

### Ejemplo 3: Con Cuantización (8-bit)

```python
from local_llm_chat import UniversalChatClient

# Transformers con cuantización 8-bit
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,  # Requiere bitsandbytes
    device="cuda"
)

response = client.infer("Hola, ¿cómo estás?")
print(response)
```

### Ejemplo 4: Cambio Dinámico de Backend

```python
from local_llm_chat import UniversalChatClient

# Empezar con GGUF
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.gguf"
)

# ... usar el modelo ...

# Cambiar a Transformers
client.change_model(
    model_path="bigscience/bloom-560m",  # o usar model_name_or_path en kwargs
    backend="transformers"
)

# Alternativa: usar model_name_or_path en kwargs
client.change_model(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

# Ahora usa Transformers
response = client.infer("Nueva pregunta")
print(response)
```

## Documentación Adicional

- **Documentación Completa**: Ver `README.md`
- **Configuración**: Ver `CONFIG.md`
- **Arquitectura Multi-Backend**: Ver `doc/03.11.25/BACKENDS_ARCHITECTURE.md`
- **Ejemplos Completos**: Ver `EXAMPLES.md` (19 ejemplos)
- **Alias de Parámetros**: Ver `doc/03.11.25/PARAMETER_ALIASES.md`
- **Fix Python 3.13**: Ver `doc/03.11.25/PYTHON_3.13_FIX.md`
- **Estructura del Proyecto**: Ver `PROJECT_STRUCTURE.md`
- **Verificar Instalación**: Ejecutar `python verify_installation.py`

## Solución de Problemas

### Python 3.13 - Error con RAG

```
ImportError: cannot import name 'Sequence' from 'collections'
```

**Solución**: RAG requiere Python 3.11 o 3.12. Ver `doc/03.11.25/PYTHON_3.13_FIX.md`

```bash
# Opción 1: Usar sin RAG (Python 3.13)
pip install -r requirements.txt  # Solo core

# Opción 2: Cambiar a Python 3.11/3.12
pyenv install 3.12.0
pyenv local 3.12.0
pip install -r requirements-rag.txt
```

### El modelo GGUF no carga

```bash
# Verificar si el archivo existe
ls models/

# Intentar con un modelo diferente
# Ver README.md para recomendaciones
```

### Errores con Transformers

```bash
# Asegúrate de tener instalado el backend
pip install -e ".[transformers]"

# Para modelos grandes, usa cuantización
pip install -e ".[quantization]"
```

### Errores de importación

```bash
# Reinstalar en modo de desarrollo
pip install -e .

# Verificar instalación
python verify_installation.py
```

### Sin memoria suficiente

```bash
# Para GGUF: Descargar cuantización menor (Q4 vs Q8)
/models  # Ver recomendaciones

# Para Transformers: Usar cuantización 8-bit o descargar modelo más pequeño
/models  # Ver recomendaciones según tu RAM
/download bigscience/bloom-560m  # Modelo pequeño para pruebas

# O usar cuantización 8-bit
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="modelo",
    load_in_8bit=True  # Reduce uso de memoria ~50%
)
```

## Funcionalidades v2.0

Ahora puedes:

- ✅ Chatear con **modelos GGUF** (llama.cpp)
- ✅ Chatear con **modelos Transformers** (Hugging Face)
- ✅ **Cambiar entre backends** sin reiniciar
- ✅ Usar **system prompts adaptativos** (presets o personalizados)
- ✅ **RAG** para procesamiento de documentos (Python 3.11/3.12)
- ✅ **Cuantización 8-bit/4-bit** para ahorrar memoria
- ✅ **Aceleración GPU** (CUDA/Metal)
- ✅ Guardar y revisar conversaciones
- ✅ Usar como **biblioteca de Python**
- ✅ Compatible con **Python 3.8 - 3.13**
