# 📋 Changelog - Local LLM Chat

Registro completo de cambios y mejoras del proyecto.

---

## 📅 2025-11-04 — v2.0.3: Modelos Transformers en ./models/ por consistencia

**Archivos modificados:**
- `src/local_llm_chat/cli.py`

**Resumen:**
Los modelos Transformers ahora se descargan en `./models/` en lugar del caché de HuggingFace, manteniendo consistencia con los modelos GGUF.

**Cambio principal:**

**Antes:**
- GGUF: se descargaba en `./models/` ✓
- Transformers: se descargaba en `~/.cache/huggingface/` ✗

**Ahora:**
- GGUF: se descarga en `./models/` ✓
- Transformers: se descarga en `./models/` ✓

**Implementación:**
1. Uso de `snapshot_download()` para descargar modelos Transformers completos
2. Descarga en `./models/{org_model}/` con nombre normalizado
3. Cliente carga desde ruta local (no desde repo_id)
4. No usa caché de HuggingFace → sin duplicidades

**Beneficios:**
- ✅ Consistencia: ambos backends usan `./models/`
- ✅ Organización: todos los modelos en un solo lugar
- ✅ Portabilidad: fácil copiar/mover el directorio `./models/`
- ✅ Sin duplicidades: no se almacena en caché de HF
- ✅ Transparente: el usuario solo ejecuta `/download <num|id>`

**UX:**
```bash
> /download 12
[INFO] Downloading Transformers model to ./models/...
[SUCCESS] Model downloaded to: ./models/Qwen_Qwen2-0.5B
[SUCCESS] Model loaded successfully!
```

---

## 📅 2025-11-04 — v2.0.2: Bugfixes críticos y refactoring de arquitectura

**Archivos modificados:**
- `src/local_llm_chat/backends/transformers_backend.py`
- `src/local_llm_chat/client.py`
- `src/local_llm_chat/__init__.py`
- `pyproject.toml`

**Resumen:**
Arreglados bugs críticos identificados en revisión de código, refactorización de responsabilidades y fix de dependencia opcional `accelerate`.

**Bugs críticos corregidos:**

1. **Bug: `repeat_penalty` no soportado en TransformersBackend**
   - **Problema**: `TransformersBackend.generate()` no aceptaba `repeat_penalty`, pero `client.infer()` lo pasaba
   - **Impacto**: Parámetro ignorado silenciosamente en modelos Transformers
   - **Fix**: Añadido parámetro `repeat_penalty` con mapeo a `repetition_penalty` de HuggingFace
   - **Archivos**: `src/local_llm_chat/backends/transformers_backend.py` (líneas 249, 314)

2. **Bug: Memory leak en `change_model()`**
   - **Problema**: Al cambiar de modelo, el anterior no se descargaba de memoria
   - **Impacto**: Consumo acumulativo de RAM/VRAM en cambios frecuentes
   - **Fix**: Llamada explícita a `backend.unload_model()` antes de cargar nuevo modelo
   - **Archivos**: `src/local_llm_chat/client.py` (líneas 614-620)

**Mejoras importantes:**

3. **Validación de parámetros en `infer()`**
   - Añadidas validaciones para `prompt`, `max_tokens`, `temperature`, `top_p`, `repeat_penalty`, `top_k`
   - Mensajes de error descriptivos con valores inválidos
   - Previene errores en backends por parámetros fuera de rango
   - **Archivos**: `src/local_llm_chat/client.py` (líneas 403-445)

4. **Normalización de `device_map` en TransformersBackend**
   - **Problema**: Mezcla de strings simples ("cuda", "mps") con `device_map="auto"` de HuggingFace
   - **Fix**: Usar `device_map="auto"` de HF para balanceo inteligente cuando `device="auto"`
   - **Fix adicional**: Fallback inteligente cuando `accelerate` no está instalado
   - Si `accelerate` disponible: usa `device_map="auto"` (óptimo)
   - Si no disponible: selecciona dispositivo directamente (cuda/mps/cpu)
   - Mejora mensajes informativos de detección de hardware
   - **Archivos**: `src/local_llm_chat/backends/transformers_backend.py` (líneas 184-219)

**Refactoring de arquitectura:**

5. **Separación de responsabilidades: `ConversationManager`**
   - Nueva clase `ConversationManager` para gestión de historial y métricas
   - Responsabilidad única: tracking de conversaciones
   - `UniversalChatClient` delega gestión de historial a `ConversationManager`
   - Mantiene API pública 100% compatible (sin breaking changes)
   - Facilita testing y mantenimiento futuro
   - **Archivos**: `src/local_llm_chat/client.py` (líneas 21-82, múltiples delegaciones)

**Beneficios:**
- ✅ Consistencia entre backends: GGUF y Transformers ahora aceptan los mismos parámetros
- ✅ Sin memory leaks: modelos se descargan correctamente
- ✅ Validación robusta: errores detectados temprano con mensajes claros
- ✅ Mejor uso de GPU: device_map="auto" aprovecha balanceo de HuggingFace
- ✅ Código más mantenible: responsabilidades claramente separadas
- ✅ Sin breaking changes: API pública sin modificaciones

**Documentación actualizada:**
- README.md: Añadida nota sobre `accelerate` en instalación Transformers
- README.md: Nueva sección de troubleshooting para error de `accelerate`
- QUICKSTART.md: Explicación de qué incluye cada instalación
- Clarificado que `accelerate` es opcional pero recomendado

**Testing:**
- Probados todos los backends: GGUF y Transformers
- Verificados parámetros: repeat_penalty, top_k, temperature, etc.
- Comprobada limpieza de memoria en cambio de modelos
- Validadas todas las validaciones de parámetros
- Verificado fallback sin `accelerate` funciona correctamente

---

## 📅 2025-11-04 — Feature: Comando /download mejorado con soporte para IDs de HuggingFace

**Archivos modificados:**
- `src/local_llm_chat/cli.py`
- `src/local_llm_chat/model_config.py`
- `src/local_llm_chat/utils.py`

**Resumen:**
Extendido el comando `/download` para aceptar tanto números (recomendaciones) como IDs directos de HuggingFace, con detección automática del backend.

**Cambios realizados:**

1. **Comando `/download` ahora acepta dos formatos**:
   - Números (comportamiento existente): `/download 1`
   - IDs de HuggingFace (nuevo): `/download meta-llama/Llama-3.1-8B-GGUF`

2. **Detección automática de backend mejorada**:
   - Arreglado `detect_backend_type()` para detectar "GGUF" sin punto
   - Ahora reconoce correctamente repos como `bartowski/Llama-3.2-3B-Instruct-GGUF`
   - Mantiene compatibilidad con paths locales `.gguf`

3. **Soporte para ambos backends**:
   - GGUF: descarga archivo `.gguf` del repo
   - Transformers: carga directamente (auto-download)

**Casos de uso:**
```bash
# Desde recomendaciones (existente)
/download 1

# Modelo GGUF específico (nuevo)
/download bartowski/Llama-3.2-3B-Instruct-GGUF

# Modelo Transformers específico (nuevo)
/download microsoft/phi-2
/download bigscience/bloom-560m
```

**Impacto:**
- ✅ Más flexible: acceso a cualquier modelo de HuggingFace
- ✅ Retrocompatible: números siguen funcionando igual
- ✅ Sin duplicación de código: reutiliza lógica existente
- ✅ UX mejorada: menos pasos para probar modelos específicos

---

## 📅 2025-11-04 — Refactor: Eliminado hardcoding subjetivo en recomendaciones

**Archivos modificados:**
- `src/local_llm_chat/model_config.py`

**Resumen:**
Eliminado hardcoding subjetivo en el sistema de recomendaciones para usar solo métricas objetivas de la API de HuggingFace.

**Cambios realizados:**

1. **Eliminado `priority_orgs` (hardcoding subjetivo)**:
   - Antes: priorizaba manualmente organizaciones específicas (bigscience, meta-llama, etc.)
   - Ahora: usa solo `downloads` (métrica objetiva de HuggingFace)
   - Resultado: recomendaciones basadas en popularidad real, no preferencias subjetivas

2. **Creada constante `FULL_PRECISION_SIZE_MULTIPLIER`**:
   - Antes: `estimated_size = base_gb * 2` (hardcoded)
   - Ahora: `estimated_size = base_gb * FULL_PRECISION_SIZE_MULTIPLIER`
   - Mejor mantenibilidad y claridad del código

3. **Simplificado algoritmo de ordenamiento**:
   - Antes: `sort(key=lambda x: (not x['priority'], -x['downloads']))`
   - Ahora: `sort(key=lambda x: -x['downloads'])`
   - Más simple y transparente

**Impacto:**
- ✅ Sin hardcoding subjetivo
- ✅ Recomendaciones basadas en datos reales (downloads)
- ✅ Código más mantenible
- ✅ Organizaciones nuevas/emergentes se incluyen automáticamente

---

## 📅 2025-11-04 — Feature: Sistema de recomendaciones para Transformers + detección MPS

**Archivos modificados:**
- `src/local_llm_chat/model_config.py`
- `src/local_llm_chat/utils.py`
- `src/local_llm_chat/cli.py`
- `src/local_llm_chat/backends/transformers_backend.py`

**Resumen:**
Extendido el sistema de recomendaciones inteligentes para incluir modelos Transformers además de GGUF, con detección automática de hardware (incluyendo Metal/MPS en macOS).

**Cambios realizados:**

1. **Nueva función `get_transformers_recommendations()` en `model_config.py`**:
   - Consulta la API de HuggingFace para modelos populares de Transformers
   - Filtra basándose en hardware detectado (usa thresholds específicos ya que Transformers necesita más RAM)
   - Prioriza organizaciones conocidas (bigscience, meta-llama, mistralai, etc.)
   - Retorna top 10 modelos compatibles con el hardware del usuario

2. **Thresholds específicos para Transformers**:
   - < 8GB RAM: modelos tiny (500M-560M parámetros)
   - 8-16GB RAM: modelos small (1B-1.5B parámetros)
   - 16-32GB RAM: modelos medium (3B-7B parámetros)
   - > 32GB RAM: modelos large (7B-8B parámetros)

3. **Actualizado `show_available_models()` en `utils.py`**:
   - Muestra dos secciones separadas: "GGUF MODELS" y "TRANSFORMERS MODELS"
   - Numeración continua entre ambas secciones
   - Indica características de cada backend (GGUF = rápido en CPU, Transformers = más modelos disponibles)
   - Retorna lista combinada para el comando `/download`

4. **CLI actualizada para manejar ambos backends**:
   - Detecta automáticamente el tipo de backend de cada modelo recomendado
   - GGUF: descarga archivo `.gguf` explícitamente (como antes)
   - Transformers: carga directamente usando el nombre del modelo (descarga automática por HuggingFace)
   - Muestra el backend en los mensajes de descarga/carga

5. **Detección automática de Metal/MPS en TransformersBackend**:
   - Prioridad de detección: CUDA > MPS > CPU
   - Detecta Apple Silicon (Metal Performance Shaders) automáticamente
   - Usa `torch.backends.mps.is_available()` para verificar MPS
   - Selecciona dtype automáticamente según GPU disponible (float16 en GPU, float32 en CPU)
   - Mensajes informativos sobre qué GPU se detectó

**Ejemplo de uso:**
```bash
# CLI muestra ahora ambos tipos
$ python main.py
GGUF MODELS (Recommended - Fast on CPU)
  1. bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
  2. ...

TRANSFORMERS MODELS (More RAM, any HF model)
  6. bigscience/bloom-560m
  7. ...

# Descargar GGUF (índice 1-5)
> /download 1

# Cargar Transformers (índice 6+)
> /download 6
[INFO] Backend: TRANSFORMERS
[INFO] Loading Transformers model (auto-download from HuggingFace Hub)...
[TRANSFORMERS] Detected Metal (MPS) - Apple Silicon
```

**Impacto:**
- ✅ Paridad de experiencia entre GGUF y Transformers
- ✅ Usuarios no necesitan conocer nombres exactos de modelos
- ✅ Transformers ahora soporta Apple Silicon automáticamente
- ✅ Thresholds ajustados según requisitos reales de memoria
- ✅ Mismo flujo de trabajo para ambos backends

---

## 📅 2025-11-04 — Fix: Eliminadas dependencias CUDA inválidas en pyproject.toml

**Archivos modificados:**
- `pyproject.toml`

**Resumen:**
Eliminadas las dependencias opcionales `cuda` y `cuda118` que incluían `--index-url`, formato inválido según PEP 508 que causaba errores de parseo.

**Problema identificado:**
- Las dependencias opcionales `cuda` y `cuda118` (líneas 83-88) contenían `--index-url https://download.pytorch.org/whl/cu121`
- PEP 508 no permite especificar URLs de índice directamente en especificaciones de dependencias
- Esto causaba errores de parseo al instalar el paquete

**Cambios realizados:**
1. Eliminadas las dependencias opcionales `cuda` y `cuda118`
2. Añadido comentario explicativo sobre instalación manual de PyTorch con CUDA
3. `torch>=2.0.0` en `dependencies` principal sigue instalando PyTorch CPU por defecto

**Nota:**
PyTorch con CUDA debe instalarse manualmente:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
# o
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Impacto:**
- ✅ `pyproject.toml` ahora es válido según PEP 508
- ✅ Eliminados errores de parseo durante instalación
- ✅ Instalación manual de PyTorch CUDA documentada claramente

---

## 📅 2025-11-04 — Bugfix: top_k parameter en GGUFBackend

**Archivos modificados:**
- `src/local_llm_chat/backends/gguf_backend.py`

**Resumen:**
Corregido bug donde `GGUFBackend.generate()` no aceptaba el parámetro `top_k`, causando inconsistencia con otros backends y pérdida silenciosa del parámetro.

**Problema identificado:**
- `client.infer()` pasaba `top_k` explícitamente a todos los backends (línea 437)
- `TransformersBackend.generate()` aceptaba `top_k: int = 50` correctamente
- `GGUFBackend.generate()` **no** tenía `top_k` en su firma, solo `**kwargs`
- El parámetro `top_k` se perdía silenciosamente y no se pasaba a `llm.create_chat_completion()`

**Cambios realizados:**
1. Añadido `top_k: int = 40` a la firma de `GGUFBackend.generate()`
2. Actualizado docstring para documentar el parámetro
3. Pasado `top_k` a `llm.create_chat_completion()`

**Firma actualizada:**
```python
def generate(
    self, 
    messages: List[Dict[str, str]], 
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    top_k: int = 40,  # ✅ AÑADIDO
    **kwargs
) -> Dict[str, Any]:
```

**Impacto:**
- ✅ Consistencia entre backends (GGUF y Transformers)
- ✅ El parámetro `top_k` ahora se respeta correctamente
- ✅ Mejor control sobre la generación de texto
- ✅ Interfaz unificada para todos los backends

---

## 📅 2025-11-03 — Renombrado de simple.py a simple_rag_backend.py (Coherencia)

**Archivos modificados:**
- `src/local_llm_chat/rag/simple.py` → `simple_rag_backend.py` (renombrado)
- `src/local_llm_chat/rag/__init__.py`
- `src/local_llm_chat/rag/manager.py`
- `README.md`
- `PROJECT_STRUCTURE.md`
- `CONFIG.md`

**Resumen:**
Renombrado `simple.py` a `simple_rag_backend.py` para mantener coherencia con la nomenclatura del proyecto. Todos los backends ahora siguen el mismo patrón de nombres: `*_backend.py`.

**Motivación:**
- **Coherencia interna**: `raganything_backend.py` tenía sufijo, pero `simple.py` no
- **Coherencia con backends/**: `gguf_backend.py`, `transformers_backend.py` usan el mismo patrón
- **Estándar de la industria**: Django, Keras, Celery usan `*_backend.py` para implementaciones intercambiables
- **Claridad**: El nombre indica explícitamente que es un backend RAG

**Cambios realizados:**
1. Renombrado físico del archivo
2. Actualizados imports en `rag/__init__.py` y `rag/manager.py`
3. Actualizada documentación en README, PROJECT_STRUCTURE y CONFIG

**Arquitectura resultante:**
```
src/local_llm_chat/
├── backends/
│   ├── gguf_backend.py          ✓ Coherente
│   └── transformers_backend.py  ✓ Coherente
└── rag/
    ├── simple_rag_backend.py    ✓ Coherente (antes: simple.py)
    └── raganything_backend.py   ✓ Coherente
```

**Beneficios:**
- ✅ Nomenclatura consistente en todo el proyecto
- ✅ Sigue estándares de la industria (Strategy/Backend pattern)
- ✅ Más fácil de entender para nuevos desarrolladores
- ✅ Documentación actualizada

---

## 📅 2025-11-03 — Fix Imports Condicionales RAG + Mejora requirements-rag.txt

**Archivos modificados:**
- `src/local_llm_chat/rag/__init__.py`
- `requirements-rag.txt`
- `pyproject.toml`

**Resumen:**
Implementados imports condicionales para los backends RAG, siguiendo el mismo patrón profesional que `backends/__init__.py`. Esto previene errores de importación cuando las dependencias RAG opcionales no están instaladas.

**Cambios realizados:**

1. **Imports condicionales en `rag/__init__.py`**:
   - `SimpleRAG` y `RAGAnythingBackend` ahora se importan con try/except
   - Evita errores cuando chromadb, sentence-transformers o raganything no están instalados
   - Mismo patrón que el módulo `backends`

2. **Reorganización de `requirements-rag.txt`**:
   - Secciones claras: SimpleRAG (ligero) vs RAG-Anything (pesado)
   - Comentarios profesionales con instrucciones de instalación
   - Facilita instalar solo SimpleRAG sin los conflictos de magic-pdf

3. **Actualización de `pyproject.toml`**:
   - `pypdf` actualizado de 3.0.0 a 6.0.0 (consistencia)
   - Añadidos `future>=1.0.0` y `configparser>=5.0.0` a `rag-full`
   - Mejora la compatibilidad con Python 3.11/3.12

**Beneficios:**
- ✅ El proyecto funciona sin errores aunque RAG no esté instalado
- ✅ Instalación simple de SimpleRAG sin conflictos de dependencias
- ✅ Documentación clara sobre qué instalar según las necesidades
- ✅ Patrón consistente con el resto del proyecto (backends)

**Instrucciones de instalación:**
```bash
# Solo SimpleRAG (recomendado)
pip install chromadb sentence-transformers pypdf

# RAG completo (opcional, pesado)
pip install -r requirements-rag.txt
```

---

## 📅 2025-11-03 — Resolución de Conflictos de Merge

**Archivos modificados:**
- `QUICKSTART.md`
- `README.md`
- `requirements.txt`
- `src/local_llm_chat/__init__.py`
- `src/local_llm_chat/client.py`

**Resumen:**
Resueltos todos los conflictos de merge entre las ramas `develop` y `main`. Se mantuvo la versión 2.0 del proyecto con soporte completo para múltiples backends (GGUF + Transformers), preservando todas las funcionalidades avanzadas y la documentación actualizada.

**Archivos resueltos:**
- ✅ QUICKSTART.md - Mantenida versión v2.0 con documentación multi-backend
- ✅ README.md - Preservada documentación completa v2.0
- ✅ requirements.txt - Mantenidas dependencias con Transformers opcionales
- ✅ src/local_llm_chat/__init__.py - Preservadas exportaciones de backends
- ✅ src/local_llm_chat/client.py - Mantenida implementación multi-backend

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
