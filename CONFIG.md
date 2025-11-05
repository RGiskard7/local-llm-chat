# Guía de Configuración

## Resumen

Local LLM Chat utiliza un sistema de configuración centralizado que soporta múltiples fuentes de configuración con un orden de prioridad claro.

## Archivos de Configuración

### Configuración por Defecto: `src/local_llm_chat/config.json`

```json
{
  "model": {
    "n_ctx": 8192,
    "n_gpu_layers": -1,
    "verbose": false
  },
  "llm": {
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "top_k": 40
  },
  "rag": {
    "chunk_size": 150,
    "chunk_overlap": 25,
    "top_k": 1,
    "max_context_tokens": 800
  }
}
```

### Configuración Personalizada

Crea `config.local.json` en el mismo directorio para sobrescribir los valores por defecto:

```json
{
  "rag": {
    "chunk_size": 200,
    "top_k": 2
  }
}
```

## Prioridad de Configuración

1. **Parámetros del constructor** (máxima prioridad) - para uso como biblioteca
2. **Variables de entorno** - para despliegue
3. **Archivo JSON** - para configuración persistente
4. **Valores por defecto** - fallback configurado en código

## Secciones de Configuración

### Configuración del Modelo (Carga e Inicialización)

Parámetros utilizados al cargar el modelo en memoria:

| Parámetro | Tipo | Por Defecto | Descripción |
|-----------|------|-------------|-------------|
| `n_ctx` | int | 8192 | Tamaño de la ventana de contexto en tokens |
| `n_gpu_layers` | int | -1 | Capas en GPU (-1 = automático, 0 = solo CPU) |
| `verbose` | bool | false | Habilitar logs detallados de carga del modelo |

### Configuración LLM (Inferencia y Generación)

Parámetros utilizados durante la generación de texto:

| Parámetro | Tipo | Por Defecto | Descripción |
|-----------|------|-------------|-------------|
| `max_tokens` | int | 256 | Máximo de tokens a generar por respuesta |
| `temperature` | float | 0.7 | Temperatura de muestreo (0.0-2.0) |
| `top_p` | float | 0.9 | Umbral de muestreo nucleus |
| `repeat_penalty` | float | 1.1 | Factor de penalización por repetición |
| `top_k` | int | 40 | Parámetro de muestreo top-k |

### Configuración RAG (Documentos Q&A)

Parámetros para procesamiento y recuperación de documentos RAG:

| Parámetro | Tipo | Por Defecto | Descripción |
|-----------|------|-------------|-------------|
| `chunk_size` | int | 150 | Tamaño de chunk en palabras |
| `chunk_overlap` | int | 25 | Solapamiento entre chunks en palabras |
| `top_k` | int | 1 | Número de chunks a recuperar |
| `max_context_tokens` | int | 800 | Tamaño máximo de contexto en palabras |

## Ejemplos de Uso

### 1. Aplicación Standalone

Edita `config.json` directamente:

```bash
cd src/local_llm_chat
nano config.json
```

### 2. Como Biblioteca (Inyección de Dependencias)

```python
from local_llm_chat import UniversalChatClient
from local_llm_chat.config import Config
from local_llm_chat.rag import SimpleRAG  # Importado de simple_rag_backend.py

# ✅ CORRECTO: Cargar config y pasarla explícitamente
config = Config()

# Crear cliente con configuración
client = UniversalChatClient(
    model_path="model.gguf",
    n_ctx=config.model.n_ctx,
    n_gpu_layers=config.model.n_gpu_layers,
    verbose=config.model.verbose,
    llm_config=config.llm  # Para infer()
)

# Pasar config a RAG
rag = SimpleRAG(client, config=config)

# ❌ INCORRECTO: No hacer esto
# client = UniversalChatClient(model_path="model.gguf")  # Sin config
```

### 3. Configuración Personalizada

```python
# Cargar config personalizado
config = Config(config_file="my_config.json")

# O modificar programáticamente
config = Config()
config.rag.chunk_size = 200
config.llm.max_tokens = 512
config.llm.temperature = 0.1

# Pasar a cliente
client = UniversalChatClient(
    model_path="model.gguf",
    n_ctx=config.model.n_ctx,
    llm_config=config.llm
)
```

### 4. Variables de Entorno

```bash
# Configuración del modelo
export MODEL_N_CTX=4096
export MODEL_N_GPU_LAYERS=32
export MODEL_VERBOSE=true

# Configuración LLM
export LLM_MAX_TOKENS=512
export LLM_TEMPERATURE=0.2
export LLM_TOP_P=0.95
export LLM_TOP_K=50

# Configuración RAG
export RAG_CHUNK_SIZE=200
export RAG_TOP_K=2
export RAG_MAX_CONTEXT_TOKENS=1000

# Ejecutar aplicación
python -m local_llm_chat
```

### 5. Configuración por Proyecto

Crea una configuración específica del proyecto:

```bash
# Crear config
cat > config.production.json << EOF
{
  "rag": {
    "chunk_size": 300,
    "top_k": 3
  },
  "llm": {
    "max_tokens": 512,
    "temperature": 0.2
  }
}
EOF

# Usarla
python -c "
from local_llm_chat.config import Config
config = Config('config.production.json')
print(config.rag.chunk_size)  # 300
"
```

## Ajuste de Rendimiento

### Para Modelos 3B en CPU (Por Defecto)

```json
{
  "rag": {
    "chunk_size": 150,
    "top_k": 1,
    "max_context_tokens": 800
  },
  "llm": {
    "max_tokens": 256,
    "temperature": 0.1
  }
}
```

**Esperado**: 2-4 minutos por consulta

### Para Modelos Grandes en GPU

```json
{
  "rag": {
    "chunk_size": 500,
    "top_k": 3,
    "max_context_tokens": 3000
  },
  "llm": {
    "max_tokens": 1024,
    "temperature": 0.3
  }
}
```

**Esperado**: 10-30 segundos por consulta

### Para Calidad sobre Velocidad

```json
{
  "rag": {
    "chunk_size": 300,
    "chunk_overlap": 50,
    "top_k": 3,
    "max_context_tokens": 2000
  },
  "llm": {
    "max_tokens": 512,
    "temperature": 0.2
  }
}
```

## Guardar Configuración

```python
from local_llm_chat.config import Config

# Crear o modificar config
config = Config()
config.rag.chunk_size = 200
config.llm.max_tokens = 512

# Guardar en archivo
config.save_to_file("my_config.json")
```

## Mejores Prácticas

1. **Usa archivos JSON para configuraciones persistentes** entre sesiones
2. **Usa variables de entorno para despliegue** (Docker, cloud)
3. **Usa parámetros del constructor para uso como biblioteca** (control programático)
4. **Mantén `config.json` en control de versiones** (valores por defecto)
5. **Ignora configuraciones personalizadas en git** (ya están en `.gitignore`)

## Resolución de Problemas

### ¿La configuración no se carga?

```python
from local_llm_chat.config import Config
config = Config()
print(config)  # Muestra los valores cargados
```

### ¿Qué configuración se está usando?

Comprueba la salida de consola al iniciar:
```
[RAG] Using config: chunk_size=150, top_k=1
```

### Restablecer a valores por defecto

```bash
# Eliminar config personalizada
rm src/local_llm_chat/config.local.json

# O restablecer en código
config = Config()  # Siempre carga primero los valores por defecto
```

## Avanzado: Sobrescritura Programática

```python
from local_llm_chat.config import Config, ModelConfig, LLMConfig, RAGConfig

# Crear configuración completamente personalizada
custom_model = ModelConfig(
    n_ctx=4096,
    n_gpu_layers=32,
    verbose=True
)

custom_llm = LLMConfig(
    max_tokens=400,
    temperature=0.15,
    top_p=0.95,
    repeat_penalty=1.05,
    top_k=50
)

custom_rag = RAGConfig(
    chunk_size=250,
    chunk_overlap=30,
    top_k=2,
    max_context_tokens=1500
)

config = Config()
config.model = custom_model
config.llm = custom_llm
config.rag = custom_rag
```

## Migración desde Versiones Anteriores

Los valores hardcodeados anteriores han sido reemplazados por valores por defecto configurables:

| Parámetro | Anterior | Nuevo | Ubicación |
|-----------|----------|-------|-----------|
| chunk_size | 500 | 150 | config.rag.chunk_size |
| top_k | 3 | 1 | config.rag.top_k |
| max_tokens | 512 | 256 | config.llm.max_tokens |

**Acción requerida**: Ninguna - los valores por defecto están optimizados para modelos 3B en CPU
