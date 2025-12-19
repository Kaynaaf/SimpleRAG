import streamlit as st
import faiss
import numpy as np
import uuid
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class ChunkMetadata:
    
    child_id: str
    parent_id: str
    child_text: str

@dataclass
class RetrievalResult:
   
    parent_text: str
    parent_id: str
    score: float
    child_matched: str

@dataclass
class RerankedResult:
    
    parent_text: str
    parent_id: str
    retrieval_score: float
    rerank_score: float
    child_matched: str

@dataclass
class RAGState:
    
    faiss_index: Optional[faiss.IndexFlatIP] = None
    doc_store: Dict[str, str] = field(default_factory=dict)
    metadata_store: List[ChunkMetadata] = field(default_factory=list)
    embedder: Optional[SentenceTransformer] = None
    reranker: Optional[CrossEncoder] = None

@dataclass
class SplitterConfig:
    
    parent_size: int = 2000
    child_size: int = 400
    overlap: int = 50

# Implemented Parent-child chunk splitting from scratch

@dataclass
class ParentChildSplitter:
    
    config: SplitterConfig = field(default_factory=SplitterConfig)

    def _split_text(self, text: str, size: int) -> List[str]:
        
        if len(text) <= size:
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + size, text_len)
            
            
            if end < text_len:
                lookback = text.rfind(' ', start, end)
                if lookback != -1 and lookback > start:
                    end = lookback
            
            chunks.append(text[start:end].strip())
            start += size - self.config.overlap
            
           
            if start >= end: 
                start = end
                
        return chunks

    def split_document(self, raw_text: str) -> Tuple[List[str], List[ChunkMetadata], Dict[str, str]]:
        """
        Splits doc into Parents -> Children.
        Returns:
            - child_texts: For embedding
            - metadata_list: For FAISS alignment
            - parent_storage: For Document Store
        """
        parents = self._split_text(raw_text, self.config.parent_size)
        
        child_texts_for_embedding = []
        metadata_list = []
        parent_storage = {}

        for parent_text in parents:
            # Generate Parent ID
            p_id = str(uuid.uuid4())[:8]
            parent_storage[p_id] = parent_text
            
            # Create Children from this Parent
            children = self._split_text(parent_text, self.config.child_size)
            
            for child_text in children:
                c_id = str(uuid.uuid4())[:8]
                
                child_texts_for_embedding.append(child_text)
                metadata_list.append(ChunkMetadata(
                    child_id=c_id,
                    parent_id=p_id,
                    child_text=child_text
                ))

        return child_texts_for_embedding, metadata_list, parent_storage

# Retrieve Chunks and Rerank them

@dataclass
class Retriever:
    
    state: RAGState
    
    def retrieve_candidates(self, query: str, top_k: int = 5, candidate_multiplier: int = 3) -> List[RetrievalResult]:
     
        if self.state.faiss_index is None or self.state.faiss_index.ntotal == 0:
            return []

        # Embed Query
        q_embed = self.state.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(q_embed)
        # Search FAISS for more candidates than needed
        num_candidates = min(top_k * candidate_multiplier, self.state.faiss_index.ntotal)
        distances, indices = self.state.faiss_index.search(q_embed, num_candidates)
        
        # Map back to Parents with deduplication
        unique_parent_ids = set()
        results = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1: 
                continue 
            
            meta = self.state.metadata_store[idx]
            p_id = meta.parent_id
            
            # Deduplication logic
            if p_id not in unique_parent_ids:
                parent_content = self.state.doc_store.get(p_id, "")
                results.append(RetrievalResult(
                    parent_text=parent_content,
                    parent_id=p_id,
                    score=float(dist),
                    child_matched=meta.child_text
                ))
                unique_parent_ids.add(p_id)
            
        return results
    
    def rerank_results(self, query: str, results: List[RetrievalResult], top_k: int = 5) -> List[RerankedResult]:
       
        if not results or self.state.reranker is None:
            # Fallback: convert to reranked format without reranking
            return [
                RerankedResult(
                    parent_text=r.parent_text,
                    parent_id=r.parent_id,
                    retrieval_score=r.score,
                    rerank_score=0.0,
                    child_matched=r.child_matched
                )
                for r in results[:top_k]
            ]
        
        # Prepare query-document pairs for cross-encoder
        pairs = [[query, r.parent_text] for r in results]
        
        
        rerank_scores = self.state.reranker.predict(pairs)
        
     
        reranked = []
        for result, rerank_score in zip(results, rerank_scores):
            reranked.append(RerankedResult(
                parent_text=result.parent_text,
                parent_id=result.parent_id,
                retrieval_score=result.score,
                rerank_score=float(rerank_score),
                child_matched=result.child_matched
            ))
        
        
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return reranked[:top_k]



@st.cache_resource
def get_models():
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embedder, reranker

def initialize_state():
    if 'rag_state' not in st.session_state:
        embedder, reranker = get_models()
        st.session_state.rag_state = RAGState(
            embedder=embedder,
            reranker=reranker,
            faiss_index=faiss.IndexFlatIP(384)
            
        )

def ingest_file(uploaded_file):
    """Orchestrates: Upload -> MarkItDown -> Split -> Embed -> Store"""
    md = MarkItDown()
    
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Convert to Markdown
        result = md.convert(tmp_path)
        text_content = result.text_content
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return 0
    finally:
        os.remove(tmp_path)

    # Split (Parent-Child)
    splitter = ParentChildSplitter()
    child_texts, metadata, parent_map = splitter.split_document(text_content)
    
    if not child_texts:
        return 0

    # Embed Children
    embeddings = st.session_state.rag_state.embedder.encode(child_texts)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    
    # Update Global State
    st.session_state.rag_state.faiss_index.add(embeddings)
    st.session_state.rag_state.metadata_store.extend(metadata)
    st.session_state.rag_state.doc_store.update(parent_map)
    
    return len(child_texts)

# Streamlit UI

def main():
    st.set_page_config(layout="wide")
    st.title("🧩 Parent-Child RAG with Cross-Encoder Reranking")
    initialize_state()

    with st.sidebar:
        st.header("1. Ingestion")
        uploaded = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt', 'pptx'])
        if uploaded and st.button("Ingest File"):
            with st.spinner("🔄 Converting & indexing..."):
                count = ingest_file(uploaded)
            st.success(f"✅ Indexed {count} child chunks!")
            
        st.markdown("---")
        st.header("⚙️ Retrieval Settings")
        top_k = st.slider("Top Results", 1, 10, 3)
        use_reranking = st.checkbox("Enable Cross-Encoder Reranking", value=True)
        candidate_multiplier = st.slider("Candidate Multiplier", 2, 5, 3, 
                                        help="Retrieve N×top_k candidates for reranking")
        
        st.markdown("---")
        st.header("📊 Debug Stats")
        if st.session_state.rag_state.faiss_index:
            st.metric("Total Child Chunks (FAISS)", st.session_state.rag_state.faiss_index.ntotal)
            st.metric("Total Parent Docs (Storage)", len(st.session_state.rag_state.doc_store))

    # Main Chat Interface
    st.subheader("2. Query System")
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        retriever = Retriever(state=st.session_state.rag_state)
        
        # Step 1: Retrieve candidates
        with st.spinner("🔍 Retrieving candidates..."):
            candidates = retriever.retrieve_candidates(query, top_k=top_k, candidate_multiplier=candidate_multiplier)
        
        if not candidates:
            st.warning("No relevant context found.")
        else:
            # Step 2: Rerank if enabled
            if use_reranking:
                with st.spinner("🎯 Reranking with cross-encoder..."):
                    results = retriever.rerank_results(query, candidates, top_k=top_k)
                st.info(f"Found {len(candidates)} candidates → Reranked to top {len(results)} contexts")
            else:
                # Convert to reranked format without reranking
                results = [
                    RerankedResult(
                        parent_text=r.parent_text,
                        parent_id=r.parent_id,
                        retrieval_score=r.score,
                        rerank_score=0.0,
                        child_matched=r.child_matched
                    )
                    for r in candidates[:top_k]
                ]
                st.info(f"Found {len(results)} contexts (reranking disabled)")
            
            # Display results
            for i, res in enumerate(results):
                score_display = f"Rerank: {res.rerank_score:.4f} | Retrieval: {res.retrieval_score:.4f}" if use_reranking else f"Score: {res.retrieval_score:.4f}"
                
                with st.expander(f"Context #{i+1} ({score_display})", expanded=i==0):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("#### 🎯 Matched Fragment (Child)")
                        st.code(res.child_matched, language="text")
                    
                    with col2:
                        st.markdown("#### 📖 Full Context (Parent)")
                        st.markdown(res.parent_text)

if __name__ == "__main__":
    main()
