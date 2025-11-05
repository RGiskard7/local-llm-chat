"""
Interfaz de línea de comandos para el Cliente de Chat Universal
Maneja toda la interacción del usuario y procesamiento de comandos
"""

import os
import json
import asyncio
from huggingface_hub import hf_hub_download, HfApi, snapshot_download

from .client import UniversalChatClient
from .utils import list_local_models, show_available_models, show_help
from .config import Config
from .model_config import _select_preferred_gguf_file

# RAG imports - lazy loading (solo se carga cuando se usa /rag)
RAG_AVAILABLE = False
RAGManager = None

def _check_rag_availability():
    """Check if RAG dependencies are available (lazy check)"""
    global RAG_AVAILABLE, RAGManager
    if RAGManager is not None:
        return RAG_AVAILABLE
    
    try:
        from .rag import RAGManager as _RAGManager
        RAGManager = _RAGManager
        RAG_AVAILABLE = True
        return True
    except ImportError as e:
        RAG_AVAILABLE = False
        print(f"[INFO] RAG no disponible: {e}")
        print("[INFO] Instala con: pip install raganything magic-pdf[full] sentence-transformers")
        return False


def run_cli():
    """Función principal de la CLI."""
    print("\n" + "=" * 60)
    print("  UNIVERSAL CHAT CLIENT - TERMINAL INTERFACE")
    print("=" * 60)

    # Cargar configuración (RESPONSABILIDAD DEL CLI, NO DEL CLIENT)
    config = Config()

    # Obtener modelos locales
    local_models = list_local_models()

    # Inicializar cliente
    client = None

    # Si hay modelos locales, mostrar solo los locales
    if local_models:
        print("\n" + "=" * 60)
        print(f"FOUND {len(local_models)} LOCAL MODEL(S)")
        print("=" * 60)
        print("\nPlease select a model:")
        for i, model_info in enumerate(local_models, 1):
            model_name = model_info['name']
            model_type = model_info['type'].upper()
            print(f"  {i}. {model_name} [{model_type}]")
        print("=" * 60)

        choice = input("\nYour choice (number): ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(local_models):
            selected_model = local_models[int(choice) - 1]
            model_path = selected_model['path']
            model_type = selected_model['type']
            model_name = selected_model['name']
            print(f"\n[INIT] Loading: {model_name} [{model_type.upper()}]")
            try:
                # Detectar backend automáticamente según tipo
                if model_type == 'transformers':
                    client = UniversalChatClient(
                        backend="transformers",
                        model_name_or_path=model_path,
                        device="auto",
                        verbose=config.model.verbose,
                        llm_config=config.llm
                    )
                else:
                    # GGUF
                    client = UniversalChatClient(
                        model_path=model_path,
                        n_ctx=config.model.n_ctx,
                        n_gpu_layers=config.model.n_gpu_layers,
                        verbose=config.model.verbose,
                        llm_config=config.llm
                    )
            except Exception as e:
                print(f"[ERROR] Failed to initialize: {e}")
                return
        else:
            print(f"[ERROR] Invalid choice: {choice}")
            print("[INFO] Please restart and choose a valid option")
            return
    
    # Si NO hay modelos locales, mostrar recomendaciones
    else:
        print("\n[INFO] No local models found. Showing recommendations...")
        cached_recommendations = show_available_models()

        if not cached_recommendations:
            print("[ERROR] Could not fetch recommendations and no local models available")
            print("[INFO] Please check your internet connection or download a model manually")
            return

        print("\nPlease select a model to download:")
        print("Or press Enter to exit")

        choice = input("\nYour choice (number): ").strip()

        if not choice or choice.lower() in ['/exit', '/quit']:
            print("[EXIT] Goodbye!")
            return

        # Procesar selección por número
        if not choice.isdigit():
            print("[ERROR] Invalid input. Please enter a number")
            return

        idx = int(choice) - 1
        if idx < 0 or idx >= len(cached_recommendations):
            print("[ERROR] Invalid model number")
            return

        model_to_download = cached_recommendations[idx]
        backend_type = model_to_download.get('backend', 'gguf')
        
        print(f"\n[DOWNLOAD] Starting download of {model_to_download['repo_id']}")
        print(f"[INFO] Backend: {backend_type.upper()}")
        print(f"[INFO] Size: ~{model_to_download['estimated_size_gb']}GB")
        print(f"[INFO] This may take several minutes...")

        try:
            if backend_type == 'gguf':
                # GGUF: descargar archivo .gguf
                api = HfApi()
                files = api.list_repo_files(model_to_download['repo_id'])
                gguf_files = [f for f in files if f.endswith('.gguf')]

                if not gguf_files:
                    print(f"[ERROR] No GGUF files found in {model_to_download['repo_id']}")
                    return

                # Usar función reutilizable para seleccionar archivo preferido
                preferred_file = _select_preferred_gguf_file(gguf_files)
                if not preferred_file:
                    print(f"[ERROR] Could not select GGUF file")
                    return

                print(f"[INFO] Downloading file: {preferred_file}")

                downloaded_path = hf_hub_download(
                    repo_id=model_to_download['repo_id'],
                    filename=preferred_file,
                    local_dir="./models",
                )

                print(f"[SUCCESS] Model downloaded successfully!")
                print(f"[INFO] Loading model...")

                client = UniversalChatClient(
                    backend="gguf",
                    model_path=downloaded_path,
                    n_ctx=config.model.n_ctx,
                    n_gpu_layers=config.model.n_gpu_layers,
                    verbose=config.model.verbose,
                    llm_config=config.llm
                )
            
            elif backend_type == 'transformers':
                # Transformers: descargar a ./models/ y cargar desde ahí
                print(f"[INFO] Downloading Transformers model to ./models/...")
                print(f"[INFO] This may take several minutes...")
                
                # Normalizar nombre para carpeta
                model_folder = model_to_download['repo_id'].replace('/', '_')
                local_model_path = f"./models/{model_folder}"
                
                # Descargar modelo completo a ./models/
                try:
                    downloaded_path = snapshot_download(
                        repo_id=model_to_download['repo_id'],
                        local_dir=local_model_path,
                    )
                    print(f"[SUCCESS] Model downloaded to: {downloaded_path}")
                except Exception as download_error:
                    print(f"[ERROR] Failed to download model: {download_error}")
                    raise
                
                # Cargar desde ruta local
                print(f"[INFO] Loading model from local path...")
                client = UniversalChatClient(
                    backend="transformers",
                    model_name_or_path=downloaded_path,  # Ruta local, no repo_id
                    device="auto",
                    verbose=config.model.verbose,
                    llm_config=config.llm
                )
                
                print(f"[SUCCESS] Model loaded successfully!")
                print(f"[INFO] You can now chat with {model_to_download['repo_id']}")
                print(f"[INFO] Model location: {downloaded_path}")

        except Exception as e:
            print(f"[ERROR] Download/load failed: {e}")
            import traceback
            traceback.print_exc()
            return

    if client is None:
        print("[ERROR] No client initialized")
        return

    # Mostrar ayuda inicial
    show_help()

    # Caché para recomendaciones (para evitar volver a consultar en /download)
    cached_recommendations = []

    # RAG Manager - inicialización automática si hay documentos de sesiones previas
    rag_manager = None
    
    # Verificar si hay documentos cargados de sesiones anteriores
    def _check_existing_documents():
        """Verifica si hay documentos cargados de sesiones anteriores"""
        try:
            metadata_file = os.path.join("./simple_rag_data", "rag_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    docs = data.get('loaded_documents', [])
                    return len(docs) > 0, docs
        except:
            pass
        return False, []
    
    # Auto-inicializar RAG si hay documentos previos
    has_docs, doc_list = _check_existing_documents()
    if has_docs:
        if _check_rag_availability():
            try:
                print(f"\n[RAG] Found {len(doc_list)} document(s) from previous session")
                rag_manager = RAGManager(client, backend="simple", config=config)
                print("[RAG] System ready. Use '/rag on' to activate RAG mode")
            except Exception as e:
                print(f"[ERROR] RAG auto-initialization failed: {e}")

    try:
        while True:
            user_input = input("\n> ").strip()

            # Procesamiento de comandos
            if user_input.lower() in ['/exit', '/quit']:
                print("\n[EXIT] Shutting down...")
                client.save_log()
                break

            elif user_input.lower() == '/save':
                client.save_log()
                continue

            elif user_input.lower() == '/history':
                client.show_history()
                continue

            elif user_input.lower() == '/clear':
                client._conversation.clear_history()
                print("[CLEAR] History cleared")
                continue

            elif user_input.lower() == '/stats':
                client.show_stats()
                continue

            elif user_input.lower() == '/help':
                show_help()
                continue

            elif user_input.lower() == '/showsystem':
                client.show_system_prompt()
                continue

            elif user_input.lower() == '/clearsystem':
                client.clear_system_prompt()
                continue

            elif user_input.lower() == '/presets':
                client.list_presets()
                continue

            elif user_input.lower().startswith('/preset '):
                preset_name = user_input[8:].strip()
                if not preset_name:
                    print("[ERROR] Usage: /preset <name>")
                    print("[INFO] Tip: Use /presets to see available options")
                    continue
                client.load_preset(preset_name)
                continue

            elif user_input.lower().startswith('/system '):
                prompt_text = user_input[8:].strip()
                if prompt_text:
                    client.set_system_prompt(prompt_text)
                else:
                    print("[ERROR] Usage: /system <prompt text>")
                continue

            elif user_input.lower() == '/models':
                cached_recommendations = show_available_models()
                continue

            elif user_input.lower().startswith('/download '):
                download_arg = user_input[10:].strip()
                
                if not download_arg:
                    print("[ERROR] Usage: /download <number|model_id>")
                    print("[INFO] Examples:")
                    print("  /download 1                              # Download from recommendations")
                    print("  /download meta-llama/Llama-3.1-8B-GGUF  # Download specific model")
                    continue
                
                # Detectar si es número (recomendaciones) o ID de HuggingFace
                if download_arg.isdigit():
                    # Modo 1: Descargar desde recomendaciones (comportamiento actual)
                    idx = int(download_arg) - 1
                    if not cached_recommendations or idx < 0 or idx >= len(cached_recommendations):
                        print("[ERROR] Invalid model number")
                        print("[INFO] Use /models to see available models first")
                        continue
                    
                    model_info = cached_recommendations[idx]
                    repo_id = model_info['repo_id']
                    backend_type = model_info.get('backend', 'gguf')
                    estimated_size = model_info.get('estimated_size_gb', 'unknown')
                    
                    print(f"\n[DOWNLOAD] Model: {repo_id}")
                    print(f"[INFO] Backend: {backend_type.upper()}")
                    print(f"[INFO] Estimated size: ~{estimated_size}GB")
                
                else:
                    # Modo 2: Descargar modelo específico por ID de HuggingFace
                    from .model_config import detect_backend_type
                    
                    repo_id = download_arg
                    backend_type = detect_backend_type(repo_id)
                    
                    print(f"\n[DOWNLOAD] Model: {repo_id}")
                    print(f"[INFO] Detected backend: {backend_type.upper()}")
                
                print(f"[INFO] This may take several minutes...")
                
                try:
                    if backend_type == 'gguf':
                        # GGUF: buscar y descargar archivo .gguf
                        api = HfApi()
                        files = api.list_repo_files(repo_id)
                        gguf_files = [f for f in files if f.endswith('.gguf')]

                        if not gguf_files:
                            print(f"[ERROR] No GGUF files found in {repo_id}")
                            continue

                        # Usar función reutilizable para seleccionar archivo preferido
                        preferred_file = _select_preferred_gguf_file(gguf_files)
                        if not preferred_file:
                            print(f"[ERROR] Could not select GGUF file")
                            continue

                        print(f"[INFO] Downloading file: {preferred_file}")

                        downloaded_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=preferred_file,
                            local_dir="./models",
                        )

                        print(f"[SUCCESS] Model downloaded successfully!")
                        print(f"[INFO] Saved to: {downloaded_path}")
                        print(f"[INFO] Use: /changemodel {os.path.relpath(downloaded_path, '.')}")
                    
                    elif backend_type == 'transformers':
                        # Transformers: descargar a ./models/ y cargar desde ahí
                        print(f"[INFO] Downloading Transformers model to ./models/...")
                        print(f"[INFO] This may take several minutes...")
                        
                        # Normalizar nombre para carpeta
                        model_folder = repo_id.replace('/', '_')
                        local_model_path = f"./models/{model_folder}"
                        
                        # Descargar modelo completo a ./models/
                        try:
                            downloaded_path = snapshot_download(
                                repo_id=repo_id,
                                local_dir=local_model_path,
                            )
                            print(f"[SUCCESS] Model downloaded to: {downloaded_path}")
                        except Exception as download_error:
                            print(f"[ERROR] Failed to download model: {download_error}")
                            raise
                        
                        # Cargar desde ruta local
                        print(f"[INFO] Loading model from local path...")
                        client = UniversalChatClient(
                            backend="transformers",
                            model_name_or_path=downloaded_path,  # Ruta local, no repo_id
                            device="auto",
                            verbose=config.model.verbose,
                            llm_config=config.llm
                        )
                        
                        print(f"[SUCCESS] Model loaded successfully!")
                        print(f"[INFO] You can now chat with {repo_id}")
                        print(f"[INFO] Model location: {downloaded_path}")

                except Exception as e:
                    print(f"[ERROR] Download/load failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"[INFO] You may need to download manually from:")
                    print(f"[INFO] https://huggingface.co/{repo_id}")

                continue

            elif user_input.lower().startswith('/changemodel '):
                model_path = user_input[13:].strip()
                if not model_path:
                    print("[ERROR] Usage: /changemodel <path>")
                    print("[INFO] Tip: Use /models to see available models")
                    continue

                # Normalizar ruta - agregar models/ si no está especificado
                if not model_path.startswith('./') and not model_path.startswith('models'):
                    # Verificar si el archivo existe tal cual primero
                    if not os.path.exists(model_path):
                        # Intentar con el prefijo models/
                        model_path = f"models/{model_path}"

                client.change_model(model_path)
                continue
            
            # Comandos RAG profesionales
            elif user_input.lower().startswith('/load '):
                # Inicializar RAG solo cuando se necesita (lazy loading)
                if rag_manager is None:
                    if not _check_rag_availability():
                        continue
                    
                    try:
                        print("\n[RAG] Initializing SimpleRAG (FAST - no knowledge graph)...")
                        print(f"[RAG] Using config: chunk_size={config.rag.chunk_size}, top_k={config.rag.top_k}")
                        rag_manager = RAGManager(client, backend="simple", config=config)
                        print("[RAG] System ready!")
                    except Exception as e:
                        print(f"[ERROR] RAG initialization failed: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                file_path = user_input[6:].strip()
                if not file_path:
                    print("[ERROR] Usage: /load <file_path>")
                    print("[INFO] Example: /load document.pdf")
                    continue
                
                rag_manager.load_document(file_path)
                continue
            
            elif user_input.lower().startswith('/unload '):
                if rag_manager is None:
                    print("[RAG] RAG not initialized. Use /load <file> first.")
                    continue
                
                file_path = user_input[8:].strip()
                if not file_path:
                    print("[ERROR] Usage: /unload <file_path>")
                    continue
                
                rag_manager.unload_document(file_path)
                continue
            
            elif user_input.lower() == '/list':
                if rag_manager is None:
                    print("[RAG] RAG not initialized. No documents loaded.")
                    continue
                
                documents = rag_manager.list_documents()
                if documents:
                    print(f"[RAG] Loaded documents ({len(documents)}):")
                    for doc in documents:
                        print(f"  - {os.path.basename(doc)}")
                else:
                    print("[RAG] No documents loaded")
                continue
            
            elif user_input.lower() == '/clear':
                if rag_manager is None:
                    print("[RAG] No documents to clear")
                    continue
                
                rag_manager.clear_all_documents()
                print("[RAG] All documents cleared")
                continue
            
            elif user_input.lower() == '/rag on':
                # Si RAG no está inicializado, intentar inicializar si hay documentos
                if rag_manager is None:
                    has_docs, doc_list = _check_existing_documents()
                    if has_docs:
                        if not _check_rag_availability():
                            continue
                        try:
                            print(f"\n[RAG] Initializing with {len(doc_list)} existing document(s)...")
                            rag_manager = RAGManager(client, backend="simple", config=config)
                            print("[RAG] System ready!")
                        except Exception as e:
                            print(f"[ERROR] RAG initialization failed: {e}")
                            continue
                    else:
                        print("[RAG] No documents loaded. Use /load <file> first.")
                        continue
                
                # Verificar que hay documentos cargados
                if not rag_manager.current_document:
                    print("[RAG] No documents loaded. Use /load <file> first.")
                    continue
                
                rag_manager.rag_mode = True
                print("[RAG] ✓ RAG mode activated - will search in documents")
                continue
            
            elif user_input.lower() == '/rag off':
                if rag_manager is None:
                    print("[RAG] RAG not initialized")
                    continue
                
                rag_manager.rag_mode = False
                print("[RAG] ✓ RAG mode deactivated - using base knowledge only")
                continue
            
            elif user_input.lower() == '/status':
                if rag_manager is None:
                    print("[RAG] RAG not initialized")
                    continue
                    
                status = rag_manager.get_status()
                print(f"[RAG] Backend: {status.get('backend')}")
                print(f"[RAG] Mode: {'ON' if status.get('rag_mode') else 'OFF'}")
                print(f"[RAG] Documents loaded: {status.get('document_count', 0)}")
                if status.get('documents'):
                    for doc in status['documents']:
                        print(f"  - {os.path.basename(doc)}")
                print(f"[RAG] Working dir: {status.get('working_dir')}")
                continue

            elif not user_input:
                continue

            # Generar respuesta (con RAG si está activo)
            print(f"\n[{client.model_type.upper()}]", end=" ", flush=True)
            try:
                # Si RAG está activo Y hay documentos cargados, usar RAG
                if rag_manager and rag_manager.rag_mode and rag_manager.current_document:
                    # ARQUITECTURA CORRECTA: RAG → Context → LLM
                    
                    # 1. Buscar contexto relevante (sin LLM)
                    rag_result = rag_manager.search_context(user_input, top_k=config.rag.top_k)
                    
                    # 2. Verificar si hay contexto útil
                    if rag_result and rag_result.get("contexts"):
                        # Construir prompt con contexto
                        contexts = rag_result["contexts"]
                        sources = rag_result.get("sources", [])
                        scores = rag_result.get("relevance_scores", [])
                        
                        # Mostrar info de retrieval
                        print(f"[RAG] {len(contexts)} chunks retrieved (scores: {[f'{s:.2f}' for s in scores[:3]]})")
                        
                        # Formatear contexto
                        context_str = "\n\n---\n\n".join(contexts)
                        
                        # Limitar contexto según configuración
                        context_words = context_str.split()
                        if len(context_words) > config.rag.max_context_tokens:
                            context_str = " ".join(context_words[:config.rag.max_context_tokens]) + "..."
                            print(f"[RAG] Context limited to {config.rag.max_context_tokens} words")
                        
                        # Construir prompt RAG
                        rag_prompt = f"""Basándote EXCLUSIVAMENTE en el siguiente contexto del documento, responde la pregunta del usuario.

                        Si la información no está en el contexto, di claramente "No encuentro esa información en el documento".

                        CONTEXTO DEL DOCUMENTO:
                        {context_str}

                        PREGUNTA: {user_input}

                        RESPUESTA:"""
                        
                        # 3. Llamar al LLM con el contexto
                        # Los parámetros se toman de config automáticamente
                        response = client.infer(rag_prompt)
                        print(response)
                    else:
                        # Sin contexto relevante, respuesta normal con disclaimer
                        print("[RAG] No relevant context found. Responding without context...")
                        response = client.infer(user_input)
                        print(response)
                else:
                    # Respuesta normal sin RAG (rag_mode OFF o sin documentos)
                    response = client.infer(user_input)
                    print(response)
            except Exception as e:
                print(f"\n[ERROR] Generation failed: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Received shutdown signal")
        client.save_log()

    print("\n[SHUTDOWN] Session terminated\n")
