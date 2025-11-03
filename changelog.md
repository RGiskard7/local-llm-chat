# 📋 Changelog - Local LLM Chat

Registro completo de cambios y mejoras del proyecto.

---

## 📅 2025-11-03 — Fix Compatibilidad Python 3.13 + Actualización Docs v2.0.1

**Archivos modificados:**
- `requirements.txt`
- `requirements-rag.txt` (nuevo)
- `README.md`
- `QUICKSTART.md` (actualizado a v2.0)
- `changelog.md`

### 🔧 **Fix: Incompatibilidad RAG con Python 3.13**

**Problema identificado**:
```
ImportError: cannot import name 'Sequence' from 'collections'
```

La cadena de dependencias `raganything → lightrag-hku → future<1.0` es incompatible con Python 3.13, ya que el paquete `future` antiguo intenta importar `Sequence` desde `collections` en lugar de `collections.abc`.

**Solución implementada**:

1. **Dependencias RAG separadas**:
   - Comentadas en `requirements.txt` principal
   - Creado `requirements-rag.txt` específico
   - Core del proyecto (GGUF + Transformers) funciona en Python 3.13

2. **Documentación clara**:
   - Advertencia en `README.md` sobre versiones Python
   - Instrucciones específicas para instalar RAG
   - Badge actualizado indicando compatibilidad

3. **Ruta de migración**:
   - Python 3.13: Core + GGUF + Transformers ✅
   - Python 3.11/3.12: Todo incluyendo RAG ✅
   - RAG disponible cuando `lightrag-hku` se actualice

**Instalación RAG ahora**:
```bash
# Solo si tienes Python 3.11 o 3.12
pip install -r requirements-rag.txt
```

**Beneficios**:
- ✅ No bloquea usuarios de Python 3.13
- ✅ Core del proyecto completamente funcional
- ✅ RAG disponible en versiones anteriores
- ✅ Documentación clara de limitaciones

### 📚 **Actualización: QUICKSTART.md a v2.0**

**Problema**: `QUICKSTART.md` estaba desactualizado (v1.x), no reflejaba los cambios de v2.0.

**Cambios implementados**:

1. **Sección de requisitos actualizada**:
   - Python 3.8-3.13 (core)
   - Python 3.11-3.12 (RAG)
   - Advertencias claras sobre limitaciones

2. **Instalación por niveles**:
   - Básica (solo GGUF)
   - Completa (GGUF + Transformers)
   - Con RAG (Python 3.11/3.12)

3. **Ejemplos actualizados**:
   - ✅ Uso con backend GGUF
   - ✅ Uso con backend Transformers
   - ✅ Cuantización 8-bit
   - ✅ Cambio dinámico de backends

4. **Solución de problemas ampliada**:
   - Fix Python 3.13
   - Errores Transformers
   - Problemas de memoria
   - Guía de cuantización

5. **Comandos CLI actualizados**:
   - `/changemodel` con soporte multi-backend
   - Ejemplos con modelos HuggingFace
   - Gestión de backends

6. **Referencias actualizadas**:
   - Links a nueva documentación v2.0
   - `EXAMPLES.md`, `MIGRATION_v2.md`, `BACKENDS_ARCHITECTURE.md`
   - Fix Python 3.13

**Resultado**:
`QUICKSTART.md` ahora es una guía completa y actualizada para v2.0, con ejemplos prácticos de ambos backends y soluciones a problemas comunes.

---

## 📅 2025-11-02 — Alias de Parámetros v2.0.1

**Archivos modificados:**
- `src/local_llm_chat/client.py`
- `README.md`
- `doc/PARAMETER_ALIASES.md` (nuevo)

### 📝 **Mejora de Usabilidad: Alias model_path ↔ model_name_or_path**

**Problema identificado**:
- Backend GGUF usaba `model_path`
- Backend Transformers usaba `model_name_or_path`
- Esto requería recordar dos nombres diferentes según el backend

**Solución implementada**:
Ambos parámetros ahora son **completamente intercambiables** con cualquier backend:

```python
# GGUF - Ambas formas funcionan
client = UniversalChatClient(backend="gguf", model_path="models/llama.gguf")
client = UniversalChatClient(backend="gguf", model_name_or_path="models/llama.gguf")

# Transformers - Ambas formas funcionan
client = UniversalChatClient(backend="transformers", model_name_or_path="bigscience/bloom")
client = UniversalChatClient(backend="transformers", model_path="bigscience/bloom")
```

**Características**:
- ✅ Validación: Error claro si intentas usar ambos a la vez
- ✅ Documentación: Guía completa en `doc/PARAMETER_ALIASES.md`
- ✅ Flexibilidad: Usa el nombre que prefieras
- ✅ Convención: Respeta convenciones de ambas librerías
- ✅ Compatibilidad: Código existente funciona sin cambios

**Recomendaciones** (pero ambos son válidos):
- GGUF → `model_path` (más específico para archivos locales)
- Transformers → `model_name_or_path` (más descriptivo para nombres HF)

**Beneficios**:
- Mayor flexibilidad sin confusión
- Código más intuitivo según contexto
- Consistencia con convenciones originales de cada librería
- Sin breaking changes

---

## 📅 2025-11-02 — Sistema Multi-Backend v2.0.0 🎉

### 🚀 **NUEVA CARACTERÍSTICA MAYOR: Sistema Multi-Backend**

**Archivos creados:**
- `src/local_llm_chat/backends/__init__.py`
- `src/local_llm_chat/backends/base.py`
- `src/local_llm_chat/backends/gguf_backend.py`
- `src/local_llm_chat/backends/transformers_backend.py`
- `doc/BACKENDS_ARCHITECTURE.md`

**Archivos modificados:**
- `src/local_llm_chat/client.py` (refactorización completa)
- `src/local_llm_chat/__init__.py`
- `src/local_llm_chat/model_config.py`
- `README.md`
- `requirements.txt`
- `pyproject.toml`

### 📝 **Cambios Implementados**

#### 1. **Arquitectura Modular de Backends**

**Nueva jerarquía**:
```
ModelBackend (Abstract Interface)
    ├─> GGUFBackend (llama-cpp-python)
    └─> TransformersBackend (Hugging Face)
```

**Interfaz común** (`base.py`):
```python
class ModelBackend(ABC):
    def load_model() -> bool
    def generate(messages, max_tokens, ...) -> dict
    def unload_model()
    def get_model_info() -> dict
    def format_messages(messages, system_prompt) -> list
    @property is_loaded -> bool
```

**Ventajas**:
- ✅ Intercambiabilidad total entre backends
- ✅ Fácil agregar nuevos backends (vLLM, ONNX, etc.)
- ✅ Testing independiente por backend
- ✅ Sistema de prompts universal

#### 2. **GGUFBackend - Backend Original Refactorizado**

**Archivo**: `gguf_backend.py`

Migración de toda la lógica GGUF desde `UniversalChatClient` al backend dedicado:
- Carga de modelos .gguf locales
- Detección automática de tipo de modelo
- GPU automática (CUDA/Metal)
- System prompts adaptativos

**Compatibilidad**: 100% compatible con código existente

#### 3. **TransformersBackend - NUEVO**

**Archivo**: `transformers_backend.py`

Backend completamente nuevo para modelos Hugging Face:
- ✅ Modelos remotos desde HuggingFace Hub
- ✅ Modelos locales (PyTorch/SafeTensors)
- ✅ Multi-arquitectura (GPT, Llama, Mistral, BERT, Bloom, Falcon, etc.)
- ✅ Cuantización 8-bit/4-bit (bitsandbytes)
- ✅ Chat templates automáticos
- ✅ GPU automática con accelerate
- ✅ System prompts adaptativos

**Ejemplos**:
```python
# Modelo remoto
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

# Modelo local
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="/path/to/model",
    device="cuda"
)

# Con cuantización
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    load_in_8bit=True
)
```

#### 4. **UniversalChatClient Refactorizado**

**Cambios mayores**:
- Ahora es un orquestador de backends (no contiene lógica de inferencia)
- Constructor con parámetro `backend` ("gguf" o "transformers")
- Método `change_model()` soporta cambio de backend
- Interfaz pública sin cambios (compatibilidad hacia atrás)

**Ejemplo de cambio dinámico**:
```python
# Iniciar con GGUF
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b.gguf"
)

# Cambiar a Transformers
client.change_model(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)
```

#### 5. **Detección Automática de Backend**

**Nuevas funciones en `model_config.py`**:
```python
detect_backend_type(model_identifier: str) -> str
is_gguf_model(model_identifier: str) -> bool
is_transformers_model(model_identifier: str) -> bool
```

**Lógica de detección**:
- Si termina en `.gguf` → "gguf"
- Si contiene `/` (nombre HF) → "transformers"
- Si es directorio con `config.json` → "transformers"
- Default → "gguf" (compatibilidad)

#### 6. **Sistema de Dependencias Modular**

**Dependencias opcionales** (`pyproject.toml`):
```toml
[project.optional-dependencies]
transformers = ["transformers>=4.35.0", "accelerate>=0.20.0"]
quantization = ["transformers>=4.35.0", "accelerate>=0.20.0", "bitsandbytes>=0.41.0"]
rag = ["chromadb>=0.5.0", "sentence-transformers>=2.2.0", "pypdf>=3.0.0"]
all = [...]  # Todo incluido
```

**Instalación modular**:
```bash
pip install -e .                        # Solo GGUF
pip install -e ".[transformers]"        # + Transformers
pip install -e ".[quantization]"        # + cuantización
pip install -e ".[all]"                 # Todo
```

#### 7. **Compatibilidad con RAG**

**Ambos backends funcionan con RAG** sin cambios:
```python
# Funciona con GGUF
client = UniversalChatClient(backend="gguf", ...)
rag = RAGManager(client, backend="simple")

# Funciona con Transformers
client = UniversalChatClient(backend="transformers", ...)
rag = RAGManager(client, backend="simple")
```

#### 8. **Documentación Completa**

**Nuevo archivo**: `doc/BACKENDS_ARCHITECTURE.md`
- Explicación detallada de la arquitectura
- Ejemplos de uso para ambos backends
- Comparación GGUF vs Transformers
- Guía de instalación
- Troubleshooting

**README actualizado**:
- Sección "Backends Soportados"
- Ejemplos de uso para ambos backends
- Tabla comparativa
- Instrucciones de instalación modular

### 💡 **Beneficios de la Refactorización**

| Aspecto | Antes (v1.x) | Ahora (v2.0) |
|---------|--------------|--------------|
| **Backends** | Solo GGUF | GGUF + Transformers |
| **Arquitectura** | Monolítico | Modular |
| **Cambio de modelo** | Solo GGUF | Entre backends |
| **Extensibilidad** | Difícil | Fácil (interfaz común) |
| **Testing** | Acoplado | Independiente |
| **Modelos disponibles** | ~200 GGUF | Miles (HF + GGUF) |

### 🎯 **Casos de Uso Nuevos**

1. **Experimentación rápida**:
   ```python
   # Probar modelo HF sin descargar GGUF
   client = UniversalChatClient(
       backend="transformers",
       model_name_or_path="bigscience/bloom-560m"
   )
   ```

2. **Fine-tuning local**:
   ```python
   # Usar modelo custom entrenado
   client = UniversalChatClient(
       backend="transformers",
       model_name_or_path="/path/to/finetuned/model"
   )
   ```

3. **Comparación de backends**:
   ```python
   # Comparar velocidad GGUF vs Transformers
   client.change_model(backend="gguf", ...)
   # vs
   client.change_model(backend="transformers", ...)
   ```

### 📊 **Métricas de Implementación**

- **Archivos nuevos**: 5
- **Archivos modificados**: 6
- **Líneas de código**: ~1500 nuevas
- **Tests**: Backend interface validada
- **Documentación**: 2 documentos nuevos
- **Compatibilidad hacia atrás**: 100%

### 🔮 **Próximos Pasos**

Futuros backends posibles:
- vLLM Backend (inferencia ultra-rápida)
- ONNX Backend (multiplataforma)
- TensorRT Backend (NVIDIA optimizado)
- OpenAI API Backend (compatibilidad con APIs)

---

## 📅 2025-10-25 — RAG Auto-Initialization on Startup

**Changed files:**
- `src/local_llm_chat/cli.py`

**Summary:**
Fixed RAG persistence bug. RAG now auto-initializes when there are documents from previous sessions, eliminating the need to manually `/load` already-loaded documents.

**Problem:**
- User loads document in Session 1
- Document persists to ChromaDB + metadata
- User opens Session 2
- `/rag on` fails with "use /load first" even though document is already loaded

**Solution:**
- Check for existing documents on startup
- Auto-initialize RAGManager if documents found
- Allow `/rag on` to initialize RAG if documents exist
- Show clear status messages

**New Behavior:**
```
Session 1: /load doc.pdf → Document saved
Session 2: [Startup] → "Found 1 document from previous session"
Session 2: /rag on → "RAG mode activated" (works immediately)
```

**Benefits:**
- True document persistence across sessions
- No need to re-load existing documents
- Intuitive RAG workflow
- Clear user feedback

---

## 📅 2025-10-25 — Centralized Configuration System

**Changed files:**
- `src/local_llm_chat/config.py` (new)
- `src/local_llm_chat/config.json` (new)
- `src/local_llm_chat/rag/simple.py`
- `src/local_llm_chat/rag/raganything_backend.py`
- `src/local_llm_chat/rag/manager.py`
- `src/local_llm_chat/cli.py`
- `.gitignore`

**Summary:**
Implemented professional centralized configuration system using dataclasses + JSON. All RAG and LLM parameters are now configurable via code, JSON files, or environment variables.

**🎯 Configuration System:**

1. **Config Module** (`config.py`)
   - `RAGConfig`: chunk_size, chunk_overlap, top_k, max_context_tokens
   - `LLMConfig`: max_tokens, temperature, top_p, repeat_penalty
   - `Config`: Main class with hybrid loading strategy

2. **Loading Priority** (highest to lowest)
   - Constructor parameters (for library usage)
   - Environment variables (for deployment)
   - JSON file (for persistent config)
   - Default values (hardcoded)

3. **Default Configuration** (`config.json`)
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

**Benefits:**
- ✅ **Centralized**: One place for all config
- ✅ **Flexible**: Code, JSON, or env vars
- ✅ **Library-friendly**: Constructor parameters
- ✅ **Deployment-ready**: Environment variables
- ✅ **Optimized**: Fast defaults for 3B models on CPU
- ✅ **Professional**: Standard industry pattern

---

## 📅 2025-10-24 — Document Persistence: RAG Sessions Survive Restarts

**Changed files:**
- `src/local_llm_chat/rag/simple.py`
- `src/local_llm_chat/rag/raganything_backend.py`

**Summary:**
Implemented document persistence across sessions using dual strategy: metadata.json files + database reconstruction fallback. Documents now persist between application restarts.

**🔄 Persistence Strategy:**

1. **Metadata File** (`rag_metadata.json`)
   - Saves list of loaded documents
   - Last updated timestamp
   - Backend type and document count
   - Fast and reliable primary method

2. **Database Reconstruction** (Fallback)
   - Extracts document list from ChromaDB/Knowledge Graph
   - Automatic if metadata file is missing/corrupted
   - Ensures data is never lost

**Benefits:**
- ✅ **Zero data loss**: Documents persist across sessions
- ✅ **Automatic recovery**: Works even if metadata is lost
- ✅ **Fast startup**: Instant restoration from metadata
- ✅ **User-friendly**: No manual reload required
- ✅ **Robust**: Dual-strategy ensures reliability

---

[Entradas anteriores continúan igual...]

---

*Formato del changelog*:
- 📅 Fecha
- 🐛 Bugs corregidos
- ✨ Nuevas características
- 🔧 Mejoras
- 📝 Documentación
- 🗂️ Archivos modificados
- 💡 Contexto/razón
