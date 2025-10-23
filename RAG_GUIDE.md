# 🚀 Guía RAG - Sistema Modular

## Arquitectura

Este proyecto usa una **arquitectura modular** que permite integrar diferentes sistemas RAG fácilmente.

### Backend Actual: RAG-Anything (Configuración Profesional)

[RAG-Anything](https://github.com/HKUDS/RAG-Anything) es un framework completo configurado con máximas capacidades:

**✨ Procesamiento Multimodal Completo:**
- **MinerU Parser** con modo `auto` (máxima calidad)
- **OCR avanzado** para texto en imágenes y documentos escaneados
- **Extracción de tablas** con preservación de estructura
- **Extracción de ecuaciones** en formato LaTeX
- **Análisis de layout** profesional con modelos ML
- **LightRAG**: Sistema RAG de grafos de conocimiento
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

## 📦 Instalación

```bash
# Activar entorno virtual
source venv/bin/activate

# Instalar RAG-Anything completo (ya instalado)
pip install raganything

# Nota: Se descargarán modelos ML (~700MB) en el primer uso
# Modelos incluyen: PDF-Extract-Kit para OCR y análisis de layout
```

### ⚠️ Requisitos de Sistema

**Primera ejecución:**
- **Espacio en disco**: ~2GB libre (modelos ML + cache)
- **Memoria RAM**: 4GB mínimo recomendado
- **Primera carga**: Los modelos ML se descargan automáticamente
- **Cache**: Los modelos se almacenan en `~/.cache/huggingface/`

## 🎯 Uso

### 1. Iniciar el chat

```bash
python main.py
```

### 2. Cargar documento

```bash
> /rag /ruta/al/documento.pdf
```

**Formatos soportados:**
- PDF
- Office: DOC, DOCX, PPT, PPTX, XLS, XLSX
- Imágenes: JPG, PNG, BMP, TIFF, GIF, WebP
- Texto: TXT, MD

### 3. Hacer preguntas

```bash
> ¿Cuál es el contenido principal del documento?
> Resume los puntos clave
> ¿Qué información hay sobre X?
```

**El sistema RAG automáticamente:**
1. Encuentra los chunks relevantes del documento
2. Crea un contexto enriquecido
3. Genera respuesta basada en el documento

### 4. Ver estado

```bash
> /ragstatus
[RAG] Backend: RAG-Anything
[RAG] Documento: tu_documento.pdf
[RAG] Working dir: ./rag_data
```

## 🔧 Cómo Funciona

### Arquitectura del Sistema

```
Usuario → CLI → RAGManager → RAGAnythingBackend → LightRAG
                                                  ↓
                                            SentenceTransformers
                                                  ↓
                                         UniversalChatClient (LLM)
```

### Flujo de Carga de Documento

1. **Parsing**: MinerU extrae texto, imágenes, tablas del documento
2. **Chunking**: Divide en fragmentos de ~1024 tokens
3. **Embeddings**: Genera vectores con Sentence Transformers
4. **Indexación**: Almacena en LightRAG con grafos de conocimiento

### Flujo de Query

1. **Pregunta del usuario** → Genera embedding
2. **Búsqueda semántica** → Encuentra chunks relevantes
3. **Construcción de contexto** → Agrupa información relevante
4. **LLM + Contexto** → Genera respuesta informada

## 🔌 Añadir Otros Backends RAG

La arquitectura modular permite agregar backends fácilmente:

### Ejemplo: Añadir LlamaIndex

```python
# En rag_integration.py

from abc import ABC, abstractmethod

class LlamaIndexBackend(RAGBackend):
    """Backend usando LlamaIndex"""
    
    def __init__(self, client, working_dir="./rag_data"):
        from llama_index import VectorStoreIndex, SimpleDirectoryReader
        self.client = client
        # ... implementación
    
    def load_document(self, file_path: str) -> bool:
        # Implementar con LlamaIndex
        pass
    
    def query(self, question: str) -> str:
        # Implementar con LlamaIndex
        pass
    
    def get_status(self) -> dict:
        return {"backend": "LlamaIndex", ...}

# Registrar en RAGManager
RAGManager.AVAILABLE_BACKENDS["llamaindex"] = LlamaIndexBackend
```

### Uso:

```python
# En CLI o código
rag_manager = RAGManager(client, backend="llamaindex")
```

## 📊 Backends Disponibles

| Backend | Estado | Características |
|---------|--------|----------------|
| **RAG-Anything** | ✅ Activo | Multimodal, MinerU, LightRAG |
| LlamaIndex | 🔜 Planeado | Versatilidad, múltiples stores |
| LangChain | 🔜 Planeado | Chains, agents, tools |
| Custom | 🔧 Extensible | Implementa `RAGBackend` |

## 🛠️ Configuración Avanzada

### Cambiar Modelo de Embeddings

Edita `rag_integration.py`:

```python
# Línea ~30
self.embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
```

**Opciones:**
- `all-MiniLM-L6-v2` (384 dim, rápido) ← Actual
- `all-mpnet-base-v2` (768 dim, más preciso)
- `paraphrase-multilingual-MiniLM-L12-v2` (multilingüe)

### Configurar Parsing

El sistema está configurado con **máxima calidad** por defecto:

```python
# Configuración actual en RAGAnythingBackend.load_document()
await self.rag.process_document_complete(
    file_path=file_path,
    parse_method="auto",     # Detección automática + OCR
    display_stats=True,      # Mostrar estadísticas
    parser="mineru",         # Parser profesional MinerU
    device="cpu",            # CPU (cambiar a "cuda" si tienes GPU)
    lang="en",               # Idioma para OCR ("es" para español)
)
```

**Opciones de parse_method:**
- `"auto"` ← **Actual**: Máxima calidad, OCR completo, análisis ML
- `"ocr"`: Solo OCR intensivo
- `"txt"`: Extracción básica de texto (más rápido, sin ML)

## 📁 Estructura de Archivos

```
local-llm-chat/
├── rag_integration.py       # Sistema RAG modular
├── rag_anything_integration.py  # (Deprecado, usar rag_integration.py)
├── rag_data/                # Datos persistentes del RAG
│   ├── graph_chunk_entity_relation.graphml
│   ├── vdb_*.json
│   └── kv_store_*.json
├── requirements.txt         # Incluye raganything
└── RAG_GUIDE.md            # Esta guía
```

## 🐛 Troubleshooting

### Error: "RAG dependencies not installed"

```bash
pip install raganything magic-pdf[full]
```

### Error: Asyncio event loop

Ya está resuelto con `asyncio.run()` en la implementación actual.

### Error: "No matching distribution"

Asegúrate de estar en el entorno virtual:

```bash
source venv/bin/activate
which python  # Debe mostrar ruta al venv
```

### El documento no se procesa

Verifica formatos soportados y que el archivo existe:

```bash
ls -lh /ruta/al/archivo.pdf
```

## 📚 Referencias

- [RAG-Anything GitHub](https://github.com/HKUDS/RAG-Anything)
- [LightRAG](https://github.com/HKUDS/LightRAG)
- [Sentence Transformers](https://www.sbert.net/)
- [MinerU](https://github.com/HKUDS/MinerU)

## 🎉 Características Futuras

- [ ] Soporte para más backends (LlamaIndex, LangChain)
- [ ] UI web con Gradio
- [ ] Múltiples documentos simultáneos
- [ ] Búsqueda híbrida (vector + keyword)
- [ ] Caché de queries
- [ ] Export de respuestas a MD/PDF

---

**¿Dudas?** Abre un issue o consulta la documentación de RAG-Anything.

