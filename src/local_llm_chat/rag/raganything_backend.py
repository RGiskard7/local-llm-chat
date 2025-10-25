"""
RAG-Anything Backend - Knowledge Graph con LightRAG

Implementación compleja usando knowledge graph, multimodal, y LLM para extracción.
Ideal para modelos grandes en GPU y documentos complejos.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from .base import RAGBackend


class RAGAnythingBackend(RAGBackend):
    """
    Backend usando RAG-Anything framework con knowledge graph
    
    Características:
    - Knowledge graph con entidades y relaciones
    - Multimodal (imágenes, tablas, ecuaciones)
    - LLM para extracción de entidades
    - Parsing avanzado con MinerU
    
    Rendimiento:
    - Indexación: 20-100 minutos (múltiples llamadas LLM)
    - Query: 5-10 segundos (con knowledge graph)
    
    NOTA: Este backend incluye LLM integrado para extracción de entidades.
    Es más lento pero mucho más preciso para queries complejas.
    """
    
    def __init__(self, client, working_dir: str = "./rag_data"):
        self.client = client
        self.working_dir = working_dir
        self.metadata_file = os.path.join(working_dir, "rag_metadata.json")
        self.rag = None
        
        # Lock para proteger acceso concurrente al modelo LLM (llama-cpp no es thread-safe)
        import asyncio
        self.llm_lock = asyncio.Lock()
        
        os.makedirs(working_dir, exist_ok=True)
        
        try:
            from raganything import RAGAnything, RAGAnythingConfig
            from lightrag.utils import EmbeddingFunc
            from sentence_transformers import SentenceTransformer
            
            print("[RAG-Anything] Initializing...")
            
            # Configuración de RAG-Anything (solo working_dir es soportado)
            config = RAGAnythingConfig(
                working_dir=working_dir
            )
            
            # Modelo de embeddings
            self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            # Funciones para LightRAG
            # IMPORTANTE: Tanto LLM como embedding deben ser ASYNC para no bloquear el event loop
            
            async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                """
                Función LLM asíncrona con Lock para evitar acceso concurrente
                CRÍTICO: llama-cpp-python NO es thread-safe, necesita serialización
                """
                import asyncio
                import time
                loop = asyncio.get_event_loop()
                
                # LOCK crítico: solo 1 inferencia a la vez para evitar segfault
                async with self.llm_lock:
                    start_time = time.time()
                    print(f"[RAG-LLM] Processing prompt ({len(prompt)} chars)...")
                    
                    try:
                        # Ejecutar inferencia con parámetros OPTIMIZADOS para RAG
                        def sync_infer():
                            messages = self.client._build_messages_with_system(prompt)
                            
                            # Parámetros optimizados para extracción rápida
                            out = self.client.llm.create_chat_completion(
                                messages=messages,
                                max_tokens=512,      # Suficiente para entidades/relaciones
                                temperature=0.1,     # Más determinístico
                                top_p=0.9,
                                repeat_penalty=1.1,
                                stop=["\n\n\n"],    # Detener en líneas vacías múltiples
                            )
                            return out["choices"][0]["message"]["content"]
                        
                        result = await loop.run_in_executor(None, sync_infer)
                        elapsed = time.time() - start_time
                        print(f"[RAG-LLM] ✓ Completed in {elapsed:.1f}s ({len(result)} chars)")
                        return result
                    
                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"[RAG-LLM] Error after {elapsed:.1f}s: {e}")
                        import traceback
                        traceback.print_exc()
                        # Retornar respuesta vacía en caso de error
                        return ""
            
            # Embedding func - DEBE ser async para evitar "can't be used in 'await' expression"
            async def async_embedding_func(texts):
                """Función de embedding asíncrona requerida por LightRAG"""
                import asyncio
                import numpy as np
                
                try:
                    # Logging para ver progreso
                    num_texts = len(texts) if isinstance(texts, list) else 1
                    print(f"[RAG-EMB] Generating embeddings for {num_texts} text(s)...")
                    
                    # Ejecutar encode en un executor para no bloquear el event loop
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.embed_model.encode(
                            texts,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                    )
                    
                    print(f"[RAG-EMB] ✓ Embeddings generated (shape: {embeddings.shape})")
                    return embeddings
                
                except Exception as e:
                    print(f"[RAG-EMB] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Retornar embedding vacío del tamaño correcto
                    if isinstance(texts, list):
                        return np.zeros((len(texts), 384))
                    return np.zeros((1, 384))
            
            embedding_func = EmbeddingFunc(
                embedding_dim=384,
                max_token_size=512,
                func=async_embedding_func
            )
            
            # Inicializar RAG-Anything
            self.rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func
            )
            
            print("[RAG-Anything] System ready")
            
            # Cargar documentos desde metadata
            self._loaded_documents = self._load_documents_metadata()
            
            if self._loaded_documents:
                print(f"[RAG-Anything] ✓ Restored {len(self._loaded_documents)} document(s) from previous session")
            
        except ImportError as e:
            print(f"[RAG-Anything ERROR] Dependencies not installed: {e}")
            print("[INFO] Install with: pip install raganything magic-pdf[full] sentence-transformers")
            raise
    
    @property
    def current_document(self) -> Optional[str]:
        """Documento actual cargado en el sistema (primer documento si hay múltiples)"""
        return self._loaded_documents[0] if self._loaded_documents else None
    
    def _load_documents_metadata(self) -> List[str]:
        """
        Carga documentos desde metadata
        
        RAG-Anything usa un knowledge graph unificado, así que solo
        necesitamos cargar la lista de documentos (el grafo ya existe en disco)
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    docs = data.get('loaded_documents', [])
                    if docs:
                        print(f"[RAG-Anything] Loading {len(docs)} document(s) from metadata...")
                        return docs
            except Exception as e:
                print(f"[RAG-Anything] Could not load metadata: {e}")
        return []
    
    def _save_metadata(self):
        """
        Guarda metadata de documentos a disco
        
        Persiste la lista de documentos en el knowledge graph
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'loaded_documents': self._loaded_documents,
                    'last_updated': datetime.now().isoformat(),
                    'backend': 'RAG-Anything',
                    'document_count': len(self._loaded_documents),
                    'knowledge_graph': 'unified'
                }, f, indent=2)
        except Exception as e:
            print(f"[RAG-Anything WARNING] Could not save metadata: {e}")
    
    def load_document(self, file_path: str) -> bool:
        """
        Procesa un documento con RAG-Anything
        Patrón oficial: asyncio.run() simple
        
        NOTA: RAG-Anything usa un knowledge graph único, así que múltiples documentos
        se integran en el mismo grafo (diferente a SimpleRAG que mantiene separación).
        """
        if not os.path.exists(file_path):
            print(f"[RAG ERROR] File not found: {file_path}")
            return False
        
        # Verificar si ya está cargado
        if file_path in self._loaded_documents:
            print(f"[RAG-Anything] Document already loaded: {os.path.basename(file_path)}")
            return True
        
        try:
            print(f"[RAG] Processing document: {file_path}")
            
            import asyncio
            
            # Patrón oficial de RAG-Anything
            async def process():
                await self.rag.process_document_complete(
                    file_path=file_path,
                    output_dir=self.working_dir,
                    parse_method="auto",
                    display_stats=True
                )
            
            asyncio.run(process())
            
            # Añadir a la lista de documentos cargados
            self._loaded_documents.append(file_path)
            self._save_metadata()  # Persistir cambios
            print(f"[RAG SUCCESS] Document ready: {os.path.basename(file_path)}")
            print(f"[RAG-Anything] Total documents in knowledge graph: {len(self._loaded_documents)}")
            return True
            
        except Exception as e:
            print(f"[RAG ERROR] Failed to process: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_document(self, file_path: str) -> bool:
        """
        Remueve un documento del knowledge graph
        
        ADVERTENCIA: RAG-Anything usa un knowledge graph unificado.
        Remover un documento requiere reconstruir todo el grafo.
        Esta es una operación costosa y no recomendada.
        
        Args:
            file_path: Ruta al documento a remover
            
        Returns:
            False siempre (no soportado eficientemente)
        """
        print("[RAG-Anything WARNING] Document removal not supported efficiently")
        print("[INFO] RAG-Anything uses a unified knowledge graph")
        print("[INFO] To remove documents, use /clear and reload desired documents")
        return False
    
    def list_documents(self) -> List[str]:
        """
        Lista todos los documentos cargados en el knowledge graph
        
        Returns:
            Lista de rutas de documentos cargados
        """
        return self._loaded_documents.copy()
    
    def clear_all_documents(self) -> bool:
        """
        Limpia todos los documentos del knowledge graph
        
        ADVERTENCIA: Requiere eliminar y reinicializar el knowledge graph.
        
        Returns:
            True si exitoso
        """
        try:
            import shutil
            
            # Eliminar el directorio de trabajo completo
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
            
            # Reinicializar RAG-Anything
            from raganything import RAGAnything, RAGAnythingConfig
            config = RAGAnythingConfig(working_dir=self.working_dir)
            
            # Necesitamos recrear las funciones LLM y embedding (ya están definidas en __init__)
            # Por simplicidad, vamos a marcar que el RAG necesita reinicio
            print("[RAG-Anything WARNING] Knowledge graph cleared")
            print("[INFO] RAG-Anything instance needs reinitialization")
            print("[INFO] Restart the application to use RAG-Anything again")
            
            self._loaded_documents.clear()
            self._save_metadata()  # Persistir cambios
            return True
            
        except Exception as e:
            print(f"[RAG ERROR] Failed to clear: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_context(self, question: str, **kwargs) -> Dict:
        """
        Busca contexto relevante usando RAG-Anything
        Busca en TODOS los documentos del knowledge graph
        
        NOTA: RAG-Anything tiene LLM integrado, pero aquí solo retornamos el contexto.
        Para queries completas con knowledge graph, usar query() directamente.
        """
        if not self._loaded_documents:
            print("[RAG ERROR] No documents loaded")
            return {"contexts": [], "sources": [], "relevance_scores": []}
        
        try:
            import asyncio
            
            # RAG-Anything retorna respuesta completa, pero podemos obtener contexto
            # usando el modo "naive" que es más parecido a vector search
            async def do_search():
                return await self.rag.aquery(question, mode="naive")
            
            result = asyncio.run(do_search())
            
            # RAG-Anything retorna texto completo, lo tratamos como un contexto
            return {
                "contexts": [result] if result else [],
                "sources": [{"documents": [os.path.basename(d) for d in self._loaded_documents]}],
                "relevance_scores": [1.0]  # RAG-Anything no expone scores directamente
            }
            
        except Exception as e:
            print(f"[RAG ERROR] Failed in search_context: {e}")
            import traceback
            traceback.print_exc()
            return {"contexts": [], "sources": [], "relevance_scores": []}
    
    def query(self, question: str) -> str:
        """
        Consulta completa usando RAG-Anything con knowledge graph (LEGACY)
        Busca en TODOS los documentos del knowledge graph
        
        NOTA: Este método incluye LLM. Para arquitectura desacoplada,
        usar search_context() + LLM externo en su lugar.
        """
        if not self._loaded_documents:
            print("[RAG ERROR] No documents loaded")
            return None
        
        try:
            import asyncio
            
            # Patrón oficial de RAG-Anything con knowledge graph
            async def do_query():
                return await self.rag.aquery(question, mode="hybrid")
            
            result = asyncio.run(do_query())
            return result
            
        except Exception as e:
            print(f"[RAG ERROR] Failed in query: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_status(self) -> Dict:
        """Estado del RAG"""
        return {
            "backend": "RAG-Anything",
            "document": self.current_document,  # Primer documento
            "documents": self._loaded_documents.copy(),  # Todos los documentos
            "document_count": len(self._loaded_documents),
            "working_dir": self.working_dir,
            "knowledge_graph": "unified"
        }

