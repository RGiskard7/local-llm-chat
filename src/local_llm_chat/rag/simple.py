"""
SimpleRAG - Backend RAG rápido con ChromaDB

Implementación simple usando vectorstore sin knowledge graph.
Ideal para modelos pequeños en CPU.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from .base import RAGBackend
from ..config import Config


class SimpleRAG(RAGBackend):
    """
    Backend RAG simple y rápido - solo vectorstore, sin knowledge graph
    
    Características:
    - Chunking simple con overlap
    - Embeddings con sentence-transformers
    - Vectorstore persistente con ChromaDB
    - Sin llamadas al LLM durante indexación
    
    Rendimiento:
    - Indexación: 5-15 segundos
    - Query: < 1 segundo
    """
    
    def __init__(self, client=None, working_dir: str = "./simple_rag_data", config: Optional[Config] = None):
        """
        Inicializa SimpleRAG
        
        Args:
            client: DEPRECATED - No se necesita, se mantiene por compatibilidad
            working_dir: Directorio para almacenar datos de ChromaDB
            config: Configuración opcional (usa default si None)
        """
        # Client se mantiene por compatibilidad pero no se usa
        self.client = client  
        self.working_dir = working_dir
        self.metadata_file = os.path.join(working_dir, "rag_metadata.json")
        self.collection = None
        
        # Configuración (usa default si no se provee)
        self.config = config or Config()
        
        os.makedirs(working_dir, exist_ok=True)
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            print("[SimpleRAG] Initializing...")
            
            # Modelo de embeddings
            self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            # ChromaDB local persistente
            self.chroma_client = chromadb.PersistentClient(path=working_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            print("[SimpleRAG] ✓ System ready (indexing + context search)")
            
            # Cargar documentos desde metadata o reconstruir desde ChromaDB
            self._loaded_documents = self._load_or_reconstruct_documents()
            
            if self._loaded_documents:
                print(f"[SimpleRAG] ✓ Restored {len(self._loaded_documents)} document(s) from previous session")
            
        except ImportError as e:
            print(f"[SimpleRAG ERROR] Dependencies not installed: {e}")
            print("[INFO] Install with: pip install chromadb sentence-transformers pypdf")
            raise
    
    @property
    def current_document(self) -> Optional[str]:
        """Documento actualmente cargado (primer documento si hay múltiples)"""
        return self._loaded_documents[0] if self._loaded_documents else None
    
    def _load_or_reconstruct_documents(self) -> List[str]:
        """
        Carga documentos desde metadata o reconstruye desde ChromaDB
        
        Estrategia dual:
        1. Intentar cargar desde metadata.json (rápido)
        2. Si falla, reconstruir desde ChromaDB (fallback)
        """
        # Intentar cargar desde metadata
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    docs = data.get('loaded_documents', [])
                    if docs:
                        print(f"[SimpleRAG] Loading {len(docs)} document(s) from metadata...")
                        return docs
            except Exception as e:
                print(f"[SimpleRAG] Could not load metadata: {e}")
        
        # Si no hay metadata, reconstruir desde ChromaDB
        return self._reconstruct_from_db()
    
    def _reconstruct_from_db(self) -> List[str]:
        """
        Reconstruye lista de documentos desde ChromaDB
        
        Extrae file_paths únicos de los metadatos almacenados
        """
        try:
            all_data = self.collection.get()
            metadatas = all_data.get('metadatas', [])
            
            # Extraer file_paths únicos
            file_paths = set()
            for meta in metadatas:
                if 'file_path' in meta:
                    file_paths.add(meta['file_path'])
            
            docs = sorted(list(file_paths))  # Ordenar para consistencia
            if docs:
                print(f"[SimpleRAG] Reconstructed {len(docs)} document(s) from vectorstore")
            return docs
        except Exception as e:
            print(f"[SimpleRAG] Could not reconstruct from DB: {e}")
            return []
    
    def _save_metadata(self):
        """
        Guarda metadata de documentos a disco
        
        Persiste la lista de documentos cargados para futuras sesiones
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'loaded_documents': self._loaded_documents,
                    'last_updated': datetime.now().isoformat(),
                    'backend': 'SimpleRAG',
                    'document_count': len(self._loaded_documents)
                }, f, indent=2)
        except Exception as e:
            print(f"[SimpleRAG WARNING] Could not save metadata: {e}")
    
    def load_document(self, file_path: str) -> bool:
        """
        Procesa documento de forma RÁPIDA
        Solo chunking + embeddings (sin llamadas LLM)
        Soporta múltiples documentos - cada uno se añade al índice
        """
        if not os.path.exists(file_path):
            print(f"[SimpleRAG ERROR] File not found: {file_path}")
            return False
        
        # Verificar si ya está cargado
        if file_path in self._loaded_documents:
            print(f"[SimpleRAG] Document already loaded: {os.path.basename(file_path)}")
            return True
        
        try:
            print(f"[SimpleRAG] Processing: {os.path.basename(file_path)}")
            
            # Extraer texto del PDF
            text = self._extract_text(file_path)
            
            # Dividir en chunks (usando configuración)
            chunks = self._chunk_text(
                text, 
                chunk_size=self.config.rag.chunk_size,
                overlap=self.config.rag.chunk_overlap
            )
            print(f"[SimpleRAG] {len(chunks)} chunks created (size: {self.config.rag.chunk_size} words)")
            
            # Generar embeddings
            print(f"[SimpleRAG] Generating embeddings...")
            embeddings = self.embed_model.encode(
                chunks,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            # Guardar en ChromaDB
            print(f"[SimpleRAG] Saving to vectorstore...")
            doc_name = os.path.basename(file_path)
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                ids=[f"{doc_name}_chunk_{i}" for i in range(len(chunks))],
                metadatas=[{"source": doc_name, "chunk": i, "file_path": file_path} for i in range(len(chunks))]
            )
            
            # Añadir a la lista de documentos cargados
            self._loaded_documents.append(file_path)
            self._save_metadata()  # Persistir cambios
            print(f"[SimpleRAG] ✓ Ready in seconds (no LLM calls)")
            print(f"[SimpleRAG] Total documents loaded: {len(self._loaded_documents)}")
            return True
            
        except Exception as e:
            print(f"[SimpleRAG ERROR] Failed to process: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_document(self, file_path: str) -> bool:
        """
        Remueve un documento del índice RAG
        
        Args:
            file_path: Ruta al documento a remover
            
        Returns:
            True si exitoso, False si no se encuentra
        """
        if file_path not in self._loaded_documents:
            print(f"[SimpleRAG] Document not loaded: {os.path.basename(file_path)}")
            return False
        
        try:
            doc_name = os.path.basename(file_path)
            
            # Obtener todos los IDs que pertenecen a este documento
            # ChromaDB no tiene API directa para filtrar por metadata, así que obtenemos todo y filtramos
            all_ids = self.collection.get()['ids']
            doc_ids = [id for id in all_ids if id.startswith(f"{doc_name}_chunk_")]
            
            if doc_ids:
                self.collection.delete(ids=doc_ids)
                print(f"[SimpleRAG] Removed {len(doc_ids)} chunks from: {doc_name}")
            
            # Remover de la lista
            self._loaded_documents.remove(file_path)
            self._save_metadata()  # Persistir cambios
            print(f"[SimpleRAG] ✓ Document unloaded: {doc_name}")
            print(f"[SimpleRAG] Remaining documents: {len(self._loaded_documents)}")
            return True
            
        except Exception as e:
            print(f"[SimpleRAG ERROR] Failed to unload: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_documents(self) -> List[str]:
        """
        Lista todos los documentos cargados en el sistema
        
        Returns:
            Lista de rutas de documentos cargados
        """
        return self._loaded_documents.copy()
    
    def clear_all_documents(self) -> bool:
        """
        Limpia todos los documentos del índice RAG
        
        Returns:
            True si exitoso
        """
        try:
            # Obtener todos los IDs y eliminarlos
            all_ids = self.collection.get()['ids']
            if all_ids:
                self.collection.delete(ids=all_ids)
                print(f"[SimpleRAG] Cleared {len(all_ids)} chunks from vectorstore")
            
            self._loaded_documents.clear()
            self._save_metadata()  # Persistir cambios
            print("[SimpleRAG] ✓ All documents cleared")
            return True
            
        except Exception as e:
            print(f"[SimpleRAG ERROR] Failed to clear: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_context(self, question: str, top_k: Optional[int] = None, **kwargs) -> Dict:
        """
        Busca contexto relevante para una pregunta (SIN llamar al LLM)
        Busca en TODOS los documentos cargados
        
        Args:
            question: Pregunta del usuario
            top_k: Número de chunks a retornar (usa config.rag.top_k si None)
            **kwargs: Parámetros adicionales ignorados (compatibilidad)
            
        Returns:
            dict con estructura:
            {
                "contexts": List[str],          # Chunks de texto relevantes
                "sources": List[dict],          # Metadatos de fuentes
                "relevance_scores": List[float] # Scores de similitud (distancias)
            }
        """
        if not self._loaded_documents:
            print("[SimpleRAG ERROR] No documents loaded")
            return {"contexts": [], "sources": [], "relevance_scores": []}
        
        # Usar configuración si top_k no se especifica
        if top_k is None:
            top_k = self.config.rag.top_k
        
        try:
            print(f"[SimpleRAG] Searching for relevant context (top_k={top_k})...")
            
            # Embedding de la pregunta
            question_embedding = self.embed_model.encode([question])[0]
            
            # Buscar chunks más similares
            results = self.collection.query(
                query_embeddings=[question_embedding.tolist()],
                n_results=top_k
            )
            
            # Extraer resultados
            contexts = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Convertir distancias a scores de similitud (1 - distancia)
            relevance_scores = [1 - d for d in distances]
            
            print(f"[SimpleRAG] ✓ {len(contexts)} relevant chunks (scores: {[f'{s:.2f}' for s in relevance_scores]})")
            
            return {
                "contexts": contexts,
                "sources": metadatas,
                "relevance_scores": relevance_scores
            }
            
        except Exception as e:
            print(f"[SimpleRAG ERROR] Failed in search_context: {e}")
            import traceback
            traceback.print_exc()
            return {"contexts": [], "sources": [], "relevance_scores": []}
    
    def _extract_text(self, file_path: str) -> str:
        """Extrae texto de PDF o TXT"""
        if file_path.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                print("[ERROR] pypdf not installed. Install: pip install pypdf")
                raise
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError("Solo se soportan archivos .pdf y .txt")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Divide texto en chunks con overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def get_status(self) -> Dict:
        """
        Retorna estado actual del sistema RAG
        
        Returns:
            dict con información del backend
        """
        return {
            "backend": "SimpleRAG",
            "document": self.current_document,  # Primer documento
            "documents": self._loaded_documents.copy(),  # Todos los documentos
            "document_count": len(self._loaded_documents),
            "working_dir": self.working_dir,
            "chunks_stored": self.collection.count() if self.collection else 0,
            "architecture": "Decoupled (RAG → Context → LLM external)"
        }

