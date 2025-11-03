# Ejemplos de Uso - Local LLM Chat v2.0

Este documento contiene ejemplos completos de uso del sistema multi-backend.

## Tabla de Contenidos

1. [Uso Básico](#uso-básico)
2. [Backend GGUF](#backend-gguf)
3. [Backend Transformers](#backend-transformers)
4. [Cambio Dinámico de Backends](#cambio-dinámico-de-backends)
5. [System Prompts](#system-prompts)
6. [RAG con Ambos Backends](#rag-con-ambos-backends)
7. [Casos de Uso Avanzados](#casos-de-uso-avanzados)

---

## Uso Básico

### Ejemplo 1: Chat Simple con GGUF

```python
from local_llm_chat import UniversalChatClient

# Inicializar con modelo GGUF local
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b-instruct.gguf"
)

# Conversación
response = client.infer("¿Qué es la inteligencia artificial?")
print(response)

# Ver estadísticas
client.show_stats()

# Guardar sesión
client.save_log()
```

### Ejemplo 2: Chat Simple con Transformers

```python
from local_llm_chat import UniversalChatClient

# Inicializar con modelo HuggingFace
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

# Conversación
response = client.infer("¿Qué es la inteligencia artificial?")
print(response)

# Ver información del modelo
model_info = client.backend.get_model_info()
print(model_info)
```

---

## Backend GGUF

### Ejemplo 3: GGUF con GPU

```python
from local_llm_chat import UniversalChatClient

client = UniversalChatClient(
    backend="gguf",
    model_path="models/mistral-7b-instruct.gguf",
    n_gpu_layers=-1,  # Todas las capas en GPU
    system_prompt="Eres un experto en programación Python."
)

# Hacer preguntas técnicas
questions = [
    "¿Qué es un decorador en Python?",
    "Dame un ejemplo de uso de decoradores",
    "¿Cuándo debería usar decoradores?"
]

for q in questions:
    print(f"\nPregunta: {q}")
    response = client.infer(q)
    print(f"Respuesta: {response}\n")
    print("-" * 60)
```

### Ejemplo 4: GGUF con Model Key (Legacy)

```python
from local_llm_chat import UniversalChatClient

# Usar modelo predefinido
client = UniversalChatClient(
    model_key="llama-3-8b",  # Se descarga automáticamente
    system_prompt="Eres un asistente útil."
)

response = client.infer("Explícame qué son los transformers en ML")
print(response)
```

---

## Backend Transformers

### Ejemplo 5: Transformers con Modelo Pequeño

```python
from local_llm_chat import UniversalChatClient

# Modelo pequeño para pruebas rápidas
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m",
    device="auto",  # GPU si está disponible, sino CPU
    torch_dtype="auto"  # float16 en GPU, float32 en CPU
)

# Conversación multi-turno
questions = [
    "¿Qué es Python?",
    "¿Por qué es popular?",
    "Dame ejemplos de uso"
]

for q in questions:
    response = client.infer(q, max_tokens=100)
    print(f"Q: {q}")
    print(f"A: {response}\n")
```

### Ejemplo 6: Transformers con Cuantización 8-bit

```python
from local_llm_chat import UniversalChatClient, TRANSFORMERS_AVAILABLE

if not TRANSFORMERS_AVAILABLE:
    print("Transformers no instalado. Instalar con: pip install transformers accelerate")
    exit()

# Modelo grande con cuantización
# Requiere: pip install bitsandbytes
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    device="cuda",
    load_in_8bit=True,  # Cuantización 8-bit
    system_prompt="Eres un experto en inteligencia artificial."
)

response = client.infer(
    "Explícame cómo funcionan los modelos de lenguaje grandes",
    max_tokens=512
)
print(response)
```

### Ejemplo 7: Transformers con Modelo Local

```python
from local_llm_chat import UniversalChatClient

# Usar modelo entrenado localmente
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="/path/to/my/finetuned/model",
    device="cuda",
    torch_dtype="float16",
    trust_remote_code=False  # Seguridad
)

response = client.infer("Test del modelo personalizado")
print(response)
```

---

## Cambio Dinámico de Backends

### Ejemplo 8: Comparación de Backends

```python
import time
from local_llm_chat import UniversalChatClient

question = "Explícame qué es un transformer en machine learning"

# Iniciar con GGUF
print("=" * 60)
print("PROBANDO GGUF BACKEND")
print("=" * 60)

client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b-q8.gguf"
)

start = time.time()
response_gguf = client.infer(question)
time_gguf = time.time() - start

print(f"Respuesta GGUF: {response_gguf[:200]}...")
print(f"Tiempo: {time_gguf:.2f}s")

# Cambiar a Transformers
print("\n" + "=" * 60)
print("PROBANDO TRANSFORMERS BACKEND")
print("=" * 60)

client.change_model(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

start = time.time()
response_tf = client.infer(question)
time_tf = time.time() - start

print(f"Respuesta Transformers: {response_tf[:200]}...")
print(f"Tiempo: {time_tf:.2f}s")

# Comparación
print("\n" + "=" * 60)
print("COMPARACIÓN")
print("=" * 60)
print(f"GGUF:         {time_gguf:.2f}s")
print(f"Transformers: {time_tf:.2f}s")
print(f"Diferencia:   {abs(time_gguf - time_tf):.2f}s")
```

### Ejemplo 9: Cambio Durante Sesión

```python
from local_llm_chat import UniversalChatClient

# Sesión con múltiples modelos
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b.gguf",
    system_prompt="Eres un asistente de programación."
)

# Consultas con GGUF
response1 = client.infer("¿Qué es Python?")
response2 = client.infer("Dame un ejemplo de código")

# Cambiar a Transformers para comparar
client.change_model(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

# Mismas consultas con Transformers
response3 = client.infer("¿Qué es Python?")
response4 = client.infer("Dame un ejemplo de código")

# Ver historial (incluye ambos backends)
client.show_history()
```

---

## System Prompts

### Ejemplo 10: System Prompts con Presets

```python
from local_llm_chat import UniversalChatClient

client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b.gguf"
)

# Ver presets disponibles
client.list_presets()

# Cargar preset de coding
client.load_preset("coding")
response = client.infer("¿Cómo implemento un singleton en Python?")
print("CODING:", response)

# Cambiar a preset creative
client.load_preset("creative")
response = client.infer("Escribe un cuento corto sobre un robot")
print("\nCREATIVE:", response)

# Limpiar preset
client.clear_system_prompt()
response = client.infer("Hola")
print("\nSIN PRESET:", response)
```

### Ejemplo 11: System Prompts Personalizados

```python
from local_llm_chat import UniversalChatClient

# Backend GGUF con prompt personalizado
client = UniversalChatClient(
    backend="gguf",
    model_path="models/mistral-7b.gguf",
    system_prompt="""Eres un tutor de matemáticas experto.
    Tu objetivo es explicar conceptos complejos de forma simple.
    Usa ejemplos prácticos y paso a paso."""
)

response = client.infer("Explícame qué es una derivada")
print(response)

# Cambiar prompt durante sesión
client.set_system_prompt("""Eres un profesor de historia.
Explicas eventos históricos de forma entretenida.""")

response = client.infer("¿Qué pasó en la Revolución Francesa?")
print(response)
```

---

## RAG con Ambos Backends

### Ejemplo 12: RAG con GGUF

```python
from local_llm_chat import UniversalChatClient
from local_llm_chat.rag import RAGManager
from local_llm_chat.config import Config

# Cliente GGUF
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b.gguf"
)

# RAG Manager
config = Config()
config.rag.chunk_size = 200
config.rag.top_k = 2

rag = RAGManager(client, backend="simple", config=config)

# Cargar documentos
rag.load_document("documento1.pdf")
rag.load_document("documento2.txt")

# Listar documentos
documents = rag.list_documents()
print(f"Documentos cargados: {documents}")

# Activar RAG
rag.rag_mode = True

# Consultar
result = rag.search_context("¿Qué dice sobre el tema X?")
print(f"Contextos encontrados: {len(result['contexts'])}")
```

### Ejemplo 13: RAG con Transformers

```python
from local_llm_chat import UniversalChatClient
from local_llm_chat.rag import RAGManager
from local_llm_chat.config import Config

# Cliente Transformers
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

# RAG Manager (mismo workflow)
config = Config()
rag = RAGManager(client, backend="simple", config=config)

# Cargar documento
rag.load_document("manual.pdf")
rag.rag_mode = True

# Hacer preguntas sobre el documento
questions = [
    "¿Cuál es el tema principal?",
    "¿Qué dice sobre instalación?",
    "Resume los puntos clave"
]

for q in questions:
    result = rag.search_context(q, top_k=3)
    # El LLM genera respuesta con el contexto
    print(f"Q: {q}")
    print(f"Chunks relevantes: {len(result['contexts'])}")
```

### Ejemplo 14: RAG con Cambio de Backend

```python
from local_llm_chat import UniversalChatClient
from local_llm_chat.rag import RAGManager

# Iniciar con GGUF
client = UniversalChatClient(
    backend="gguf",
    model_path="models/llama-3.2-3b.gguf"
)

rag = RAGManager(client, backend="simple")
rag.load_document("research.pdf")
rag.rag_mode = True

# Consultas con GGUF
result1 = rag.search_context("¿Qué dice sobre metodología?")

# Cambiar a Transformers manteniendo los documentos
client.change_model(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

# Misma consulta con Transformers
# Los documentos persisten en RAG
result2 = rag.search_context("¿Qué dice sobre metodología?")

print("GGUF chunks:", len(result1['contexts']))
print("Transformers chunks:", len(result2['contexts']))
```

---

## Casos de Uso Avanzados

### Ejemplo 15: Detección Automática de Backend

```python
from local_llm_chat.model_config import detect_backend_type

# Detectar tipo de backend automáticamente
models = [
    "models/llama-3.2-3b.gguf",
    "bigscience/bloom-560m",
    "/path/to/local/pytorch/model",
    "EleutherAI/gpt-neo-125M"
]

for model in models:
    backend = detect_backend_type(model)
    print(f"{model} → backend: {backend}")
```

### Ejemplo 16: Verificar Disponibilidad de Backends

```python
from local_llm_chat import TRANSFORMERS_AVAILABLE

if TRANSFORMERS_AVAILABLE:
    print("✓ Transformers backend disponible")
    from local_llm_chat import TransformersBackend
    print("✓ Puede usar modelos HuggingFace")
else:
    print("✗ Transformers backend no instalado")
    print("  Instalar con: pip install transformers accelerate")

# Usar backend apropiado
from local_llm_chat import UniversalChatClient

if TRANSFORMERS_AVAILABLE:
    client = UniversalChatClient(
        backend="transformers",
        model_name_or_path="bigscience/bloom-560m"
    )
else:
    client = UniversalChatClient(
        backend="gguf",
        model_path="models/llama-3.2-3b.gguf"
    )
```

### Ejemplo 17: Uso Directo de Backends

```python
from local_llm_chat.backends import GGUFBackend, TransformersBackend

# Uso directo de GGUFBackend
gguf_backend = GGUFBackend(
    model_path="models/llama-3.2-3b.gguf",
    n_gpu_layers=-1
)
gguf_backend.load_model()

messages = [
    {"role": "user", "content": "Hola"}
]
result = gguf_backend.generate(messages, max_tokens=100)
print(result['content'])

# Uso directo de TransformersBackend
tf_backend = TransformersBackend(
    model_name_or_path="bigscience/bloom-560m",
    device="auto"
)
tf_backend.load_model()

result = tf_backend.generate(messages, max_tokens=100)
print(result['content'])
```

### Ejemplo 18: Configuración Avanzada

```python
from local_llm_chat import UniversalChatClient
from local_llm_chat.config import Config, RAGConfig, LLMConfig

# Configuración personalizada
config = Config()
config.rag.chunk_size = 300
config.rag.top_k = 3
config.llm.max_tokens = 512
config.llm.temperature = 0.2

# Cliente con configuración
client = UniversalChatClient(
    backend="transformers",
    model_name_or_path="bigscience/bloom-560m"
)

# Generar con parámetros personalizados
response = client.infer(
    "Explícame los transformers",
    max_tokens=config.llm.max_tokens,
    temperature=config.llm.temperature
)
print(response)
```

### Ejemplo 19: Comparación de Múltiples Modelos

```python
from local_llm_chat import UniversalChatClient
import time

models = [
    {"backend": "gguf", "model_path": "models/llama-3.2-3b.gguf"},
    {"backend": "transformers", "model_name_or_path": "bigscience/bloom-560m"},
    {"backend": "transformers", "model_name_or_path": "EleutherAI/gpt-neo-125M"},
]

question = "¿Qué es la inteligencia artificial?"
results = []

for model_config in models:
    print(f"\nProbando: {model_config}")
    client = UniversalChatClient(**model_config)
    
    start = time.time()
    response = client.infer(question, max_tokens=100)
    elapsed = time.time() - start
    
    results.append({
        "model": str(model_config),
        "response": response[:100] + "...",
        "time": elapsed
    })
    
    client.backend.unload_model()

# Mostrar resultados
print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)
for r in results:
    print(f"\nModelo: {r['model']}")
    print(f"Tiempo: {r['time']:.2f}s")
    print(f"Respuesta: {r['response']}")
```

---

## Conclusión

Estos ejemplos demuestran la flexibilidad y potencia del sistema multi-backend de Local LLM Chat v2.0. Puedes:

- ✅ Usar GGUF para máxima velocidad en CPU
- ✅ Usar Transformers para acceso a cualquier modelo HF
- ✅ Cambiar entre backends durante la sesión
- ✅ Mantener la misma interfaz y workflow
- ✅ Usar RAG con ambos backends
- ✅ Combinar system prompts y configuración avanzada

Para más información, consulta:
- [README.md](README.md) - Documentación completa
- [doc/BACKENDS_ARCHITECTURE.md](doc/BACKENDS_ARCHITECTURE.md) - Arquitectura de backends
- [QUICKSTART.md](QUICKSTART.md) - Guía de inicio rápido

