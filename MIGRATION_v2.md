# Guía de Migración a v2.0 - Sistema Multi-Backend

## Resumen de Cambios

Local LLM Chat v2.0 introduce un sistema modular de backends que permite usar modelos GGUF y Transformers de forma intercambiable, manteniendo **100% de compatibilidad hacia atrás**.

## ¿Qué Cambió?

### ✅ **Lo Nuevo**

1. **Backend Transformers** - Usa cualquier modelo de HuggingFace
2. **Arquitectura Modular** - Backends intercambiables con interfaz común
3. **Cambio Dinámico** - Cambia entre GGUF y Transformers durante la sesión
4. **Instalación Modular** - Instala solo lo que necesitas
5. **Detección Automática** - El sistema detecta el tipo de backend automáticamente

### ✅ **Lo que NO Cambió (Compatibilidad)**

- La API pública de `UniversalChatClient` sigue igual
- Todos los métodos existentes funcionan igual
- CLI sin cambios
- RAG funciona igual con ambos backends
- System prompts funcionan igual
- Logs y sesiones mantienen el mismo formato

## Migración para Usuarios Existentes

### Si Solo Usas GGUF (No Cambios Necesarios)

```python
# Este código sigue funcionando exactamente igual
from local_llm_chat import UniversalChatClient

client = UniversalChatClient(
    model_path="models/llama-3.2-3b.gguf",
    system_prompt="Eres un asistente útil."
)

response = client.infer("Hola")
```

**No necesitas cambiar nada.** Tu código existente funciona sin modificaciones.

### Si Quieres Usar Transformers (Nuevo)

```bash
# 1. Instalar dependencias de Transformers
pip install transformers accelerate

# O con dependencias opcionales
pip install -e ".[transformers]"
```

```python
# 2. Usar el nuevo backend
from local_llm_chat import UniversalChatClient

client = UniversalChatClient(
    backend="transformers",  # NUEVO parámetro
    model_name_or_path="bigscience/bloom-560m"
)

response = client.infer("Hola")
```

## Nuevos Parámetros

### UniversalChatClient

#### Parámetros Nuevos

```python
UniversalChatClient(
    # NUEVO: Selección de backend
    backend="gguf",  # o "transformers"
    
    # NUEVO: Parámetros para Transformers
    model_name_or_path="bigscience/bloom-560m",
    device="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    load_in_8bit=False,
    load_in_4bit=False,
    
    # Existentes (sin cambios)
    model_key=None,
    model_path=None,
    n_gpu_layers=-1,
    system_prompt=None,
)
```

#### Parámetros Existentes (Sin Cambios)

Todos los parámetros GGUF existentes funcionan igual:
- `model_path`
- `model_key`
- `repo_id` / `filename`
- `n_gpu_layers`
- `n_ctx`
- `system_prompt`

### change_model()

Ahora acepta cambio de backend:

```python
# Antes (v1.x)
client.change_model(model_path="models/otro-modelo.gguf")

# Ahora (v2.0) - Sigue funcionando igual
client.change_model(model_path="models/otro-modelo.gguf")

# NUEVO - Cambiar a Transformers
client.change_model(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)
```

## Nuevas Funciones Disponibles

### Detección de Backend

```python
from local_llm_chat.model_config import (
    detect_backend_type,
    is_gguf_model,
    is_transformers_model
)

backend = detect_backend_type("models/llama.gguf")  # "gguf"
backend = detect_backend_type("bigscience/bloom")   # "transformers"

is_gguf = is_gguf_model("models/llama.gguf")  # True
is_tf = is_transformers_model("bigscience/bloom")  # True
```

### Verificación de Disponibilidad

```python
from local_llm_chat import TRANSFORMERS_AVAILABLE

if TRANSFORMERS_AVAILABLE:
    print("Transformers backend disponible")
else:
    print("Instalar con: pip install transformers accelerate")
```

### Acceso Directo a Backends

```python
from local_llm_chat.backends import GGUFBackend, TransformersBackend

# Uso avanzado - acceso directo a backends
backend = GGUFBackend(model_path="...")
backend = TransformersBackend(model_name_or_path="...")
```

## Instalación

### Opción 1: Solo GGUF (Como Antes)

```bash
pip install -e .
```

### Opción 2: Con Transformers

```bash
# Básico
pip install -e ".[transformers]"

# Con cuantización 8-bit/4-bit
pip install -e ".[quantization]"

# Todo incluido
pip install -e ".[all]"
```

## Casos de Uso

### Caso 1: Usar Solo GGUF (Sin Cambios)

```python
from local_llm_chat import UniversalChatClient

# Tu código existente funciona igual
client = UniversalChatClient(
    model_path="models/llama-3.2-3b.gguf"
)
```

### Caso 2: Experimentar con HuggingFace

```python
from local_llm_chat import UniversalChatClient

# Probar modelos sin descargar GGUF
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)
```

### Caso 3: Usar Ambos en el Mismo Proyecto

```python
from local_llm_chat import UniversalChatClient

# Cliente GGUF para producción (rápido)
client_prod = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b.gguf"
)

# Cliente Transformers para desarrollo (flexible)
client_dev = UniversalChatClient(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)
```

### Caso 4: Cambio Dinámico

```python
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b.gguf"
)

# Trabajar con GGUF
response1 = client.infer("Pregunta 1")

# Cambiar a Transformers
client.change_model(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

# Trabajar con Transformers
response2 = client.infer("Pregunta 2")
```

## RAG - Sin Cambios

RAG funciona exactamente igual con ambos backends:

```python
from local_llm_chat import UniversalChatClient
from local_llm_chat.rag import RAGManager

# GGUF
client = UniversalChatClient(backend="gguf", model_path="...")
rag = RAGManager(client, backend="simple")

# O Transformers - mismo código
client = UniversalChatClient(backend="transformers", model_name_or_path="...")
rag = RAGManager(client, backend="simple")

# Workflow idéntico
rag.load_document("doc.pdf")
rag.rag_mode = True
result = rag.search_context("pregunta")
```

## Troubleshooting

### Error: "Transformers backend not available"

```bash
pip install transformers accelerate
```

### Error: "bitsandbytes not available"

```bash
# Para cuantización 8-bit/4-bit
pip install bitsandbytes
```

### Verificar Instalación

```python
from local_llm_chat import TRANSFORMERS_AVAILABLE

print(f"Transformers: {TRANSFORMERS_AVAILABLE}")

if TRANSFORMERS_AVAILABLE:
    from local_llm_chat import TransformersBackend
    print("✓ Backend Transformers listo")
```

## Breaking Changes

**No hay breaking changes.** La versión 2.0 mantiene 100% de compatibilidad hacia atrás.

Todo el código existente funciona sin modificaciones. Los nuevos parámetros son opcionales.

## Beneficios de Migrar

### Si Actualizas tu Código para Usar v2.0

✅ **Acceso a Miles de Modelos**: Cualquier modelo de HuggingFace  
✅ **Experimentación Rápida**: Probar modelos sin descargar GGUF  
✅ **Fine-tuning**: Usar modelos entrenados localmente  
✅ **Cuantización Dinámica**: 8-bit/4-bit en tiempo de carga  
✅ **Flexibilidad**: Cambiar de backend según necesidad  
✅ **Futuro**: Más backends vendrán (vLLM, ONNX, etc.)  

### Si NO Actualizas tu Código

✅ **Todo Sigue Funcionando**: Sin cambios necesarios  
✅ **GGUF Sigue Siendo la Opción por Defecto**: Máxima velocidad  
✅ **Sin Nuevas Dependencias**: Solo si quieres Transformers  

## Documentación

- [README.md](README.md) - Documentación completa actualizada
- [BACKENDS_ARCHITECTURE.md](doc/BACKENDS_ARCHITECTURE.md) - Arquitectura de backends
- [EXAMPLES.md](EXAMPLES.md) - 19 ejemplos completos de uso
- [changelog.md](changelog.md) - Registro detallado de cambios

## Soporte

Si tienes problemas con la migración:

1. Verifica que tu código existente funciona (debería)
2. Para usar Transformers, instala dependencias: `pip install transformers accelerate`
3. Revisa [EXAMPLES.md](EXAMPLES.md) para casos de uso
4. Abre un issue en GitHub si encuentras problemas

## Resumen

**Para usuarios existentes**: No necesitas hacer nada, todo funciona igual.

**Para nuevos usuarios**: Puedes elegir entre GGUF (rápido) y Transformers (flexible).

**Para todos**: Ahora tienes más opciones sin perder funcionalidad existente.

---

*Versión 2.0.0 - Noviembre 2025*

