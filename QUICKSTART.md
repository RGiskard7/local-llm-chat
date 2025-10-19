# Guía de Inicio Rápido - Local LLM Chat

Instrucciones para poner en marcha Local LLM Chat en pocos minutos.

## Instalación

```bash
# 1. Clonar o navegar al proyecto
cd local-llm-chat

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# 3. Instalar el paquete
pip install -e .
```

## Primera Ejecución

### Opción A: Ya tienes un modelo GGUF

```bash
# Ejecutar la aplicación
python main.py

# Seleccionar modelo de la lista
# Comenzar a chatear
```

### Opción B: Descargar un modelo recomendado

```bash
# Ejecutar la aplicación
python main.py

# La aplicación mostrará recomendaciones basadas en tu RAM
# Seleccionar un número para descargar
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
/models             # Listar modelos disponibles
/download 1         # Descargar modelo recomendado
/changemodel <path> # Cambiar a un modelo diferente
```

### Gestión de Sesión

```bash
/save               # Guardar conversación ahora
/clear              # Limpiar historial
/stats              # Mostrar estadísticas
/exit               # Guardar y salir
```

## Uso como Biblioteca

```python
from local_llm_chat import UniversalChatClient

# Crear cliente
client = UniversalChatClient(
    model_path="models/llama-3.1-8b-instruct.gguf",
    system_prompt="Eres un asistente útil."
)

# Generar respuesta
response = client.infer("¿Qué es Python?")
print(response)

# Guardar sesión
client.save_log()
```

## Documentación Adicional

- **Documentación Completa**: Ver `README.md`
- **Estructura del Proyecto**: Ver `PROJECT_STRUCTURE.md`
- **Guía de Migración**: Ver `doc/18.10.2025/MIGRATION.md`
- **Verificar Instalación**: Ejecutar `python verify_installation.py`

## Solución de Problemas

### El modelo no carga

```bash
# Verificar si el archivo existe
ls models/

# Intentar con un modelo diferente
# Ver README.md para recomendaciones
```

### Errores de importación

```bash
# Reinstalar en modo de desarrollo
pip install -e .
```

### Sin memoria

```bash
# Usar /models para ver recomendaciones
# Descargar una cuantización más pequeña (Q4 en lugar de Q8)
```

## Funcionalidades

Ahora puedes:

- Chatear con cualquier modelo GGUF
- Usar system prompts preconfigurados
- Cambiar modelos durante la sesión
- Guardar y revisar conversaciones
- Usar como biblioteca de Python
