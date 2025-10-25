# Estructura del Proyecto Local LLM Chat

## Árbol de Directorios Completo

```
local-llm-chat/
│
├── src/local_llm_chat/       # Paquete principal
│   ├── __init__.py              # API del paquete (exporta UniversalChatClient, Config, etc.)
│   ├── __main__.py              # Punto de entrada para `python -m local_llm_chat`
│   ├── client.py                # Clase UniversalChatClient (lógica de negocio principal)
│   ├── cli.py                   # Interfaz CLI y procesamiento de comandos
│   ├── utils.py                 # Funciones de utilidad (descubrimiento de modelos, ayuda)
│   ├── model_config.py          # Detección y configuración de modelos
│   ├── prompts.py               # Presets de system prompts
│   ├── config.py                # Sistema de configuración centralizada
│   ├── config.json              # Configuración por defecto (RAG y LLM)
│   └── rag/                     # Módulo RAG
│       ├── __init__.py          # API del módulo RAG
│       ├── base.py              # Clase abstracta RAGBackend
│       ├── simple.py            # SimpleRAG (ChromaDB, rápido)
│       ├── raganything_backend.py  # RAGAnything (knowledge graph, complejo)
│       └── manager.py           # RAGManager (orquestador)
│
├── tests/                    # Pruebas unitarias
│   ├── __init__.py
│   └── test_model_config.py     # Pruebas de detección de modelos
│
├── doc/                      # Documentación histórica
│   └── ...
│
├── models/                   # Modelos GGUF descargados (gitignored)
│   └── *.gguf
│
├── chat_logs/                # Registros de sesiones (gitignored)
│   └── *.json
│
├── simple_rag_data/          # Datos de SimpleRAG (gitignored)
│   ├── chroma.sqlite3           # Base de datos ChromaDB
│   └── rag_metadata.json        # Metadatos de documentos cargados
│
├── rag_data/                 # Datos de RAG-Anything (gitignored)
│   └── ...
│
├── main.py                   # Punto de entrada simple
├── pyproject.toml            # Empaquetado moderno de Python
├── requirements.txt          # Dependencias
├── .gitignore                # Reglas de git ignore
│
├── README.md                 # Documentación principal
├── PROJECT_STRUCTURE.md      # Este archivo
├── CONFIG.md                 # Guía de configuración
├── changelog.md              # Registro de cambios
├── config.example.json       # Ejemplo de configuración personalizada
│
└── verify_installation.py   # Script de verificación de instalación
```

## Responsabilidades de Archivos

### Paquete Principal (`src/local_llm_chat/`)

| Archivo | Líneas | Responsabilidad |
|------|-------|---------------|
| `client.py` | ~410 | Clase UniversalChatClient, gestión de conversaciones, carga de modelos |
| `cli.py` | ~500 | Interfaz CLI, procesamiento de comandos, interacción del usuario, RAG workflow |
| `utils.py` | ~110 | Funciones auxiliares (listado de modelos, recomendaciones, ayuda) |
| `model_config.py` | ~410 | Detección de tipo de modelo, mapeo de formato de chat, detección de hardware |
| `prompts.py` | ~50 | Presets de system prompts (coding, creative, etc.) |
| `config.py` | ~170 | Sistema de configuración centralizada (RAG y LLM) |
| `__init__.py` | ~30 | Inicialización del paquete, exportaciones de API pública |
| `__main__.py` | ~7 | Punto de entrada para ejecución de módulo |

### Módulo RAG (`src/local_llm_chat/rag/`)

| Archivo | Líneas | Responsabilidad |
|------|-------|---------------|
| `base.py` | ~80 | Clase abstracta RAGBackend, define interfaz común |
| `simple.py` | ~400 | SimpleRAG con ChromaDB, rápido para CPU |
| `raganything_backend.py` | ~400 | RAG-Anything con knowledge graph, para GPU |
| `manager.py` | ~150 | RAGManager, orquestador de backends |
| `__init__.py` | ~10 | API pública del módulo RAG |

### Puntos de Entrada

| Archivo | Propósito |
|------|---------|
| `main.py` | Ejecución tradicional (`python main.py`) |
| `__main__.py` | Ejecución de módulo (`python -m local_llm_chat`) |
| Script de punto de entrada | Comando instalado (`local-llm-chat` / `llm-chat`) |

### Desarrollo

| Archivo | Propósito |
|------|---------|
| `pyproject.toml` | Configuración de empaquetado moderno de Python |
| `requirements.txt` | Especificación de dependencias |
| `verify_installation.py` | Verificación automatizada de instalación |
| `example_usage.py` | Ejemplos de uso en modo biblioteca |
| `tests/` | Pruebas unitarias |

## Flujo de Importación

### Como Biblioteca

```python
from local_llm_chat import UniversalChatClient
                ↓
    src/local_llm_chat/__init__.py
                ↓
    src/local_llm_chat/client.py
```

### Como CLI

```bash
python main.py
       ↓
src/local_llm_chat/cli.py
       ↓
src/local_llm_chat/client.py
```

### Como Módulo

```bash
python -m local_llm_chat
          ↓
src/local_llm_chat/__main__.py
          ↓
src/local_llm_chat/cli.py
          ↓
src/local_llm_chat/client.py
```

## Beneficios de la Organización del Código

### Antes (Monolítico)

```
main.py (820 líneas)
├── Importaciones
├── Configuración
├── Clase UniversalChatClient (400 líneas)
├── Funciones de utilidad (150 líneas)
├── Interfaz CLI (270 líneas)
└── Función principal

Problemas:
- Difícil de probar
- Difícil de reutilizar
- Difícil de mantener
- Todo acoplado
```

### Después (Modular)

```
src/local_llm_chat/
├── client.py (410 líneas)        → Testeable
├── cli.py (300 líneas)            → Reemplazable
├── utils.py (110 líneas)          → Reutilizable
└── model_config.py (410 líneas)   → Independiente

Ventajas:
- Cada archivo tiene una responsabilidad única
- Fácil de probar en aislamiento
- Se puede intercambiar CLI por interfaz web
- Se puede usar como biblioteca
```

## Principios Clave de Diseño Aplicados

1. **Separación de Responsabilidades**: Lógica de negocio separada de UI
2. **Responsabilidad Única**: Cada módulo hace una cosa
3. **DRY (Don't Repeat Yourself)**: Las utilidades son compartidas
4. **Testeabilidad**: Funciones y clases puras
5. **Extensibilidad**: Fácil de agregar nuevas interfaces (API, GUI)
6. **Estándares Profesionales**: Sigue las mejores prácticas de empaquetado de Python

## Ejemplos de Uso

### Ejecución Tradicional

```bash
python main.py
```

### Ejecución de Módulo

```bash
python -m local_llm_chat
```

### Comando Instalado

```bash
# Después de: pip install -e .
local-llm-chat
# o alias corto:
llm-chat
```

### Uso como Biblioteca

```python
from local_llm_chat import UniversalChatClient

client = UniversalChatClient(
    model_path="models/llama-3.1-8b.gguf"
)
response = client.infer("Hola")
```

### API Programática

```python
from local_llm_chat import (
    UniversalChatClient,
    detect_model_type,
    get_hardware_info,
)

# Detectar modelo
model_type = detect_model_type("mi-modelo.gguf")

# Verificar hardware
hw = get_hardware_info()
print(f"RAM: {hw['ram_available_gb']}GB")

# Crear cliente
client = UniversalChatClient(model_path="...")
```

## Aseguramiento de Calidad

Todos los componentes verificados:

- El paquete importa correctamente
- La CLI se ejecuta exitosamente
- Las pruebas pasan
- Documentación completa
- Instalación verificada
- Múltiples métodos de ejecución funcionan

## Resultado

Un paquete de Python profesional, mantenible y extensible listo para:

- Desarrollo
- Pruebas
- Distribución
- Uso en producción
- Mejoras futuras (API, GUI, etc.)
