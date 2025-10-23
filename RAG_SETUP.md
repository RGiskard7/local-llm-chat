# 🔍 RAG Mode - Guía de Uso

## Instalación

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Las dependencias RAG incluyen:
- `lightrag-hku` - Sistema RAG
- `sentence-transformers` - Embeddings
- `numpy` - Procesamiento numérico

## Uso Rápido

### 1. Inicia el chat normal
```bash
python main.py
```

### 2. Carga un documento
```bash
> /rag test_document.txt
```

El sistema:
- ✅ Cargará el modelo de embeddings
- ✅ Procesará e indexará el documento
- ✅ Activará el modo RAG automáticamente

### 3. Haz preguntas sobre el documento
```bash
> ¿Cuál es la conclusión más importante?
> ¿Qué dice sobre búsqueda semántica?
> Resume las características principales
```

**Todas las preguntas posteriores usarán el contexto del documento automáticamente.**

### 4. Verifica el estado
```bash
> /ragstatus
```

### 5. Volver al modo normal
```bash
> /clear
```

## Formatos Soportados

Por ahora, el sistema soporta archivos de texto plano:
- `.txt`
- `.md`
- `.log`

## Ejemplo Completo

```bash
# Inicia la aplicación
python main.py

# Selecciona tu modelo
> 1

# Carga un documento
> /rag test_document.txt
[RAG] Cargando modelo de embeddings...
[RAG] Inicializando sistema RAG...
[RAG] Procesando: test_document.txt
[RAG SUCCESS] Documento indexado: test_document.txt

# Haz preguntas
> ¿Cuál es la conclusión más importante del documento?
[RAG] La conclusión más importante es que RAG combina...

> Resume las características principales
[RAG] Las características principales son: 1. Búsqueda semántica...

# Verifica el estado
> /ragstatus
[RAG] Documento activo: test_document.txt
```

## Notas Técnicas

- **Directorio de trabajo**: Los índices se guardan en `./rag_data/`
- **Modelo de embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **Modos de búsqueda**: hybrid (por defecto), naive, local, global
- **Persistencia**: Los documentos indexados se mantienen entre sesiones

## Troubleshooting

### Error: "RAG dependencies not installed"
```bash
pip install lightrag-hku sentence-transformers numpy
```

### Error: "Archivo no encontrado"
Usa la ruta completa o relativa al directorio actual:
```bash
> /rag /Users/tu-usuario/documento.txt
> /rag ./docs/archivo.txt
```

### El modelo no responde con el documento
Verifica que el documento se haya cargado:
```bash
> /ragstatus
```

## Ventajas del Modo RAG

✅ **Precisión**: Respuestas basadas en tu documento específico
✅ **Sin alucinaciones**: El modelo no inventa información
✅ **Contextual**: Entiende el contexto completo del documento
✅ **Trazable**: Puedes verificar de dónde viene la información

