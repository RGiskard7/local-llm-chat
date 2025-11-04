# Configuration Guide

## Overview

Local LLM Chat uses a centralized configuration system that supports multiple configuration sources with clear priority ordering.

## Configuration Files

### Default Configuration: `src/local_llm_chat/config.json`

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

### Custom Configuration

Create `config.local.json` in the same directory to override defaults:

```json
{
  "rag": {
    "chunk_size": 200,
    "top_k": 2
  }
}
```

## Configuration Priority

1. **Constructor parameters** (highest priority) - for library usage
2. **Environment variables** - for deployment
3. **JSON file** - for persistent config
4. **Default values** - hardcoded fallback

## Configuration Sections

### Model Configuration (Loading & Initialization)

Parameters used when loading the model into memory:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_ctx` | int | 8192 | Context window size in tokens |
| `n_gpu_layers` | int | -1 | GPU layers (-1 = auto, 0 = CPU only) |
| `verbose` | bool | false | Enable verbose model loading logs |

### LLM Configuration (Inference & Generation)

Parameters used during text generation:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | int | 256 | Maximum tokens to generate per response |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `repeat_penalty` | float | 1.1 | Repetition penalty factor |
| `top_k` | int | 40 | Top-k sampling parameter |

### RAG Configuration (Document Q&A)

Parameters for RAG document processing and retrieval:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 150 | Chunk size in words |
| `chunk_overlap` | int | 25 | Overlap between chunks in words |
| `top_k` | int | 1 | Number of chunks to retrieve |
| `max_context_tokens` | int | 800 | Maximum context size in words |

## Usage Examples

### 1. Standalone Application

Edit `config.json` directly:

```bash
cd src/local_llm_chat
nano config.json
```

### 2. As Library (Dependency Injection)

```python
from local_llm_chat import UniversalChatClient
from local_llm_chat.config import Config
from local_llm_chat.rag import SimpleRAG  # Imported from simple_rag_backend.py

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

### 3. Custom Configuration

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

### 3. Environment Variables

```bash
# Model configuration
export MODEL_N_CTX=4096
export MODEL_N_GPU_LAYERS=32
export MODEL_VERBOSE=true

# LLM configuration
export LLM_MAX_TOKENS=512
export LLM_TEMPERATURE=0.2
export LLM_TOP_P=0.95
export LLM_TOP_K=50

# RAG configuration
export RAG_CHUNK_SIZE=200
export RAG_TOP_K=2
export RAG_MAX_CONTEXT_TOKENS=1000

# Run application
python -m local_llm_chat
```

### 4. Per-Project Configuration

Create a project-specific config:

```bash
# Create config
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

# Use it
python -c "
from local_llm_chat.config import Config
config = Config('config.production.json')
print(config.rag.chunk_size)  # 300
"
```

## Performance Tuning

### For 3B Models on CPU (Default)

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

**Expected**: 2-4 minutes per query

### For Larger Models on GPU

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

**Expected**: 10-30 seconds per query

### For Quality over Speed

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

## Saving Configuration

```python
from local_llm_chat.config import Config

# Create or modify config
config = Config()
config.rag.chunk_size = 200
config.llm.max_tokens = 512

# Save to file
config.save_to_file("my_config.json")
```

## Best Practices

1. **Use JSON files for persistent configs** across sessions
2. **Use environment variables for deployment** (Docker, cloud)
3. **Use constructor parameters for library usage** (programmatic control)
4. **Keep `config.json` in version control** (default values)
5. **Ignore custom configs in git** (already in `.gitignore`)

## Troubleshooting

### Config not loading?

```python
from local_llm_chat.config import Config
config = Config()
print(config)  # Shows loaded values
```

### Which config is being used?

Check console output on startup:
```
[RAG] Using config: chunk_size=150, top_k=1
```

### Reset to defaults

```bash
# Remove custom config
rm src/local_llm_chat/config.local.json

# Or reset in code
config = Config()  # Always loads defaults first
```

## Advanced: Programmatic Override

```python
from local_llm_chat.config import Config, ModelConfig, LLMConfig, RAGConfig

# Create completely custom config
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

## Migration from Previous Versions

Previous hardcoded values have been replaced with configurable defaults:

| Parameter | Old | New | Location |
|-----------|-----|-----|----------|
| chunk_size | 500 | 150 | config.rag.chunk_size |
| top_k | 3 | 1 | config.rag.top_k |
| max_tokens | 512 | 256 | config.llm.max_tokens |

**Action required**: None - defaults are optimized for 3B models on CPU

