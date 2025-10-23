"""
Integración de RAG-Anything con UniversalChatClient
Arquitectura modular para soportar múltiples backends RAG
"""
import os
from abc import ABC, abstractmethod


class RAGBackend(ABC):
    """Interfaz abstracta para backends RAG - permite cambiar fácilmente"""
    
    @abstractmethod
    def load_document(self, file_path: str) -> bool:
        """Carga un documento en el sistema RAG"""
        pass
    
    @abstractmethod
    def query(self, question: str) -> str:
        """Realiza una query al RAG"""
        pass
    
    @abstractmethod
    def get_status(self) -> dict:
        """Retorna el estado actual del RAG"""
        pass


class RAGAnythingBackend(RAGBackend):
    """Backend usando RAG-Anything framework"""
    
    def __init__(self, client, working_dir="./rag_data"):
        self.client = client
        self.working_dir = working_dir
        self.current_document = None
        self.rag = None
        
        os.makedirs(working_dir, exist_ok=True)
        
        try:
            from raganything import RAGAnything, RAGAnythingConfig
            from lightrag.utils import EmbeddingFunc
            from sentence_transformers import SentenceTransformer
            
            print("[RAG-Anything] Inicializando...")
            
            # Configuración de RAG-Anything (solo working_dir es soportado)
            config = RAGAnythingConfig(
                working_dir=working_dir
            )
            
            # Modelo de embeddings
            self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            # Funciones SÍNCRONAS como en la documentación oficial
            # LightRAG las convierte a async automáticamente
            
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                """Función LLM síncrona - patrón oficial de RAG-Anything"""
                # Actualizar system prompt si se proporciona
                if system_prompt:
                    original_prompt = self.client.system_prompt
                    self.client.system_prompt = system_prompt
                    result = self.client.infer(prompt)
                    self.client.system_prompt = original_prompt
                    return result
                return self.client.infer(prompt)
            
            # Embedding func - lambda síncrona como en la documentación
            embedding_func = EmbeddingFunc(
                embedding_dim=384,
                max_token_size=512,
                func=lambda texts: self.embed_model.encode(
                    texts, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            )
            
            # Inicializar RAG-Anything
            self.rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func
            )
            
            print("[RAG-Anything] Sistema listo")
            
        except ImportError as e:
            print(f"[RAG-Anything ERROR] Dependencias no instaladas: {e}")
            print("[INFO] Instala con: pip install raganything magic-pdf[full] sentence-transformers")
            raise
    
    def load_document(self, file_path: str) -> bool:
        """
        Procesa un documento con RAG-Anything
        Patrón oficial: asyncio.run() simple
        """
        if not os.path.exists(file_path):
            print(f"[RAG ERROR] Archivo no encontrado: {file_path}")
            return False
        
        try:
            print(f"[RAG] Procesando documento: {file_path}")
            
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
            
            self.current_document = file_path
            print(f"[RAG SUCCESS] Documento listo: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            print(f"[RAG ERROR] Fallo al procesar: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def query(self, question: str) -> str:
        """Consulta usando RAG-Anything - Patrón oficial"""
        if not self.current_document:
            print("[RAG ERROR] No hay documento cargado")
            return None
        
        try:
            import asyncio
            
            # Patrón oficial de RAG-Anything
            async def do_query():
                return await self.rag.aquery(question, mode="hybrid")
            
            result = asyncio.run(do_query())
            return result
            
        except Exception as e:
            print(f"[RAG ERROR] Fallo en query: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_status(self) -> dict:
        """Estado del RAG"""
        return {
            "backend": "RAG-Anything",
            "document": self.current_document,
            "working_dir": self.working_dir
        }


class RAGManager:
    """
    Gestor principal de RAG - Arquitectura modular
    Permite cambiar entre diferentes backends RAG fácilmente
    """
    
    AVAILABLE_BACKENDS = {
        "raganything": RAGAnythingBackend,
        # Aquí se pueden añadir más backends en el futuro:
        # "llamaindex": LlamaIndexBackend,
        # "langchain": LangChainBackend,
        # etc.
    }
    
    def __init__(self, client, backend="raganything", working_dir="./rag_data"):
        self.client = client
        self.backend_name = backend
        
        if backend not in self.AVAILABLE_BACKENDS:
            raise ValueError(f"Backend '{backend}' no disponible. Opciones: {list(self.AVAILABLE_BACKENDS.keys())}")
        
        # Inicializar el backend seleccionado
        backend_class = self.AVAILABLE_BACKENDS[backend]
        self.backend = backend_class(client, working_dir)
    
    def load_document(self, file_path: str) -> bool:
        """Carga documento usando el backend activo"""
        return self.backend.load_document(file_path)
    
    def query(self, question: str) -> str:
        """Query usando el backend activo"""
        return self.backend.query(question)
    
    def get_status(self) -> dict:
        """Estado del sistema RAG"""
        return self.backend.get_status()
    
    @property
    def current_document(self):
        """Documento actual cargado"""
        return self.backend.current_document if hasattr(self.backend, 'current_document') else None

