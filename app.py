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
from google import genai
from google.genai import types


@dataclass
class ChunkMetadata:
    child_id: str
    parent_id: str
    child_text: str
    source_file: str  # Track which file the chunk came from

@dataclass
class RetrievalResult:
    parent_text: str
    parent_id: str
    score: float
    child_matched: str
    source_file: str

@dataclass
class RerankedResult:
    parent_text: str
    parent_id: str
    retrieval_score: float
    rerank_score: float
    child_matched: str
    source_file: str

@dataclass
class RAGState:
    faiss_index: Optional[faiss.IndexFlatIP] = None
    doc_store: Dict[str, str] = field(default_factory=dict)
    metadata_store: List[ChunkMetadata] = field(default_factory=list)
    embedder: Optional[SentenceTransformer] = None
    reranker: Optional[CrossEncoder] = None
    gemini_model: Optional[genai.Client] = None
    ingested_files: List[str] = field(default_factory=list)

@dataclass
class SplitterConfig:
    parent_size: int = 2000
    child_size: int = 400
    overlap: int = 50

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

    def split_document(self, raw_text: str, source_file: str) -> Tuple[List[str], List[ChunkMetadata], Dict[str, str]]:
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
            p_id = str(uuid.uuid4())[:8]
            parent_storage[p_id] = parent_text
            
            children = self._split_text(parent_text, self.config.child_size)
            
            for child_text in children:
                c_id = str(uuid.uuid4())[:8]
                
                child_texts_for_embedding.append(child_text)
                metadata_list.append(ChunkMetadata(
                    child_id=c_id,
                    parent_id=p_id,
                    child_text=child_text,
                    source_file=source_file
                ))

        return child_texts_for_embedding, metadata_list, parent_storage

@dataclass
class Retriever:
    state: RAGState
    
    def retrieve_candidates(self, query: str, top_k: int = 5, candidate_multiplier: int = 3) -> List[RetrievalResult]:
        if self.state.faiss_index is None or self.state.faiss_index.ntotal == 0:
            return []

        q_embed = self.state.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(q_embed)
        
        num_candidates = min(top_k * candidate_multiplier, self.state.faiss_index.ntotal)
        distances, indices = self.state.faiss_index.search(q_embed, num_candidates)
        
        unique_parent_ids = set()
        results = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1: 
                continue 
            
            meta = self.state.metadata_store[idx]
            p_id = meta.parent_id
            
            if p_id not in unique_parent_ids:
                parent_content = self.state.doc_store.get(p_id, "")
                results.append(RetrievalResult(
                    parent_text=parent_content,
                    parent_id=p_id,
                    score=float(dist),
                    child_matched=meta.child_text,
                    source_file=meta.source_file
                ))
                unique_parent_ids.add(p_id)
            
        return results
    
    def rerank_results(self, query: str, results: List[RetrievalResult], top_k: int = 5) -> List[RerankedResult]:
        if not results or self.state.reranker is None:
            return [
                RerankedResult(
                    parent_text=r.parent_text,
                    parent_id=r.parent_id,
                    retrieval_score=r.score,
                    rerank_score=0.0,
                    child_matched=r.child_matched,
                    source_file=r.source_file
                )
                for r in results[:top_k]
            ]
        
        pairs = [[query, r.parent_text] for r in results]
        rerank_scores = self.state.reranker.predict(pairs)
        
        reranked = []
        for result, rerank_score in zip(results, rerank_scores):
            reranked.append(RerankedResult(
                parent_text=result.parent_text,
                parent_id=result.parent_id,
                retrieval_score=result.score,
                rerank_score=float(rerank_score),
                child_matched=result.child_matched,
                source_file=result.source_file
            ))
        
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return reranked[:top_k]

def generate_response_stream(query: str, contexts: List[RerankedResult], client: genai.Client):
    """
    Generate streaming response using Gemini with retrieved parent contexts.
    Leverages parent-child chunking by using full parent contexts for generation.
    Yields text chunks as they arrive.
    """
    if not contexts:
        yield "I don't have enough information in the uploaded documents to answer that question. Please upload relevant documents first."
        return
    
    # Build context string with source attribution
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        context_parts.append(f"[Context {i} from {ctx.source_file}]:\n{ctx.parent_text}\n")
    
    context_string = "\n---\n".join(context_parts)
    
    # Create prompt that emphasizes using the parent contexts with proper citation format
    prompt = f"""You are a helpful assistant answering questions based on provided document contexts.

IMPORTANT INSTRUCTIONS:
- Answer the question using ONLY the information from the contexts below
- If the contexts don't contain enough information to answer fully, say so
- ALWAYS cite your sources using square brackets with numbers [1], [2], [3], etc. immediately after the relevant statement
- Use the context number that corresponds to where you found the information
- You can cite multiple sources like [1][2] if information comes from multiple contexts
- Be specific and detailed in your answer
- If multiple contexts provide relevant information, synthesize them coherently

CONTEXTS:
{context_string}

QUESTION: {query}

ANSWER (remember to cite sources with [1], [2], etc.):"""

    try:
        response = client.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"Error generating response: {str(e)}"

@st.cache_resource
def get_models(gemini_api_key: str = None):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    gemini_client = None
    if gemini_api_key:
        gemini_client = genai.Client(api_key=gemini_api_key)
    
    return embedder, reranker, gemini_client

def initialize_state():
    if 'rag_state' not in st.session_state:
        # Get API key from secrets or environment
        gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        
        embedder, reranker, gemini_client = get_models(gemini_key)
        st.session_state.rag_state = RAGState(
            embedder=embedder,
            reranker=reranker,
            gemini_model=gemini_client,
            faiss_index=faiss.IndexFlatIP(384)
        )
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False

def ingest_file(uploaded_file):
    """Orchestrates: Upload -> MarkItDown -> Split -> Embed -> Store"""
    md = MarkItDown()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        result = md.convert(tmp_path)
        text_content = result.text_content
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {e}")
        return 0
    finally:
        os.remove(tmp_path)

    # Split (Parent-Child)
    splitter = ParentChildSplitter()
    child_texts, metadata, parent_map = splitter.split_document(text_content, uploaded_file.name)
    
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
    st.session_state.rag_state.ingested_files.append(uploaded_file.name)
    
    return len(child_texts)

def clear_all_chunks():
    """Clear all stored data and reset the system"""
    st.session_state.rag_state.faiss_index = faiss.IndexFlatIP(384)
    st.session_state.rag_state.doc_store = {}
    st.session_state.rag_state.metadata_store = []
    st.session_state.rag_state.ingested_files = []
    st.session_state.chat_history = []

def main():
    st.set_page_config(page_title="NanoRAG", layout="wide", page_icon="🧠")
    
    initialize_state()

    # Sidebar
    with st.sidebar:
        st.title("📚 Document Manager")
        
        # API Key Input
        if not st.session_state.api_key_set:
            st.warning("⚠️ Gemini API key not configured")
            api_key = st.text_input("Enter Gemini API Key", type="password")
            if api_key and st.button("Set API Key"):
                embedder, reranker, gemini_client = get_models(api_key)
                st.session_state.rag_state.gemini_model = gemini_client
                st.session_state.api_key_set = True
                st.rerun()
        else:
            st.success("✅ Gemini API configured")
        
        st.markdown("---")
        
        # File Upload Section
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more documents",
            type=['pdf', 'docx', 'txt', 'pptx'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📥 Ingest Files", type="primary"):
            total_chunks = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                chunks = ingest_file(uploaded_file)
                total_chunks += chunks
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ Ingested {len(uploaded_files)} file(s) with {total_chunks} chunks!")
            st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        top_k = st.slider("Retrieved Contexts", 1, 10, 3)
        use_reranking = st.checkbox("Enable Reranking", value=True)
        
        st.markdown("---")
        
        # Stats
        st.header("📊 Statistics")
        if st.session_state.rag_state.faiss_index:
            st.metric("Child Chunks", st.session_state.rag_state.faiss_index.ntotal)
            st.metric("Parent Docs", len(st.session_state.rag_state.doc_store))
            st.metric("Files Ingested", len(st.session_state.rag_state.ingested_files))
            
            if st.session_state.rag_state.ingested_files:
                with st.expander("📁 Ingested Files"):
                    for f in st.session_state.rag_state.ingested_files:
                        st.text(f"• {f}")
        
        # Clear button at bottom
        st.markdown("---")
        if st.button("🗑️ Clear All Chunks", type="secondary", use_container_width=True):
            clear_all_chunks()
            st.success("✅ All chunks cleared!")
            st.rerun()

    # Main Chat Interface
    st.title("NanoRAG")
    st.caption("Ask questions about your uploaded documents")
    
    # Check if system is ready
    if st.session_state.rag_state.faiss_index.ntotal == 0:
        st.info("👈 Please upload documents in the sidebar to get started!")
    
    if not st.session_state.api_key_set:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"**[{i}] {src['file']}**")
                        preview = src['text'][:400] + "..." if len(src['text']) > 400 else src['text']
                        st.text(preview)
                        if i < len(message["sources"]):
                            st.markdown("---")

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            if st.session_state.rag_state.faiss_index.ntotal == 0:
                response = "Please upload documents first before asking questions."
                st.markdown(response)
            elif not st.session_state.api_key_set:
                response = "Please configure your Gemini API key in the sidebar."
                st.markdown(response)
            else:
                with st.spinner("🔍 Searching documents..."):
                    retriever = Retriever(state=st.session_state.rag_state)
                    candidates = retriever.retrieve_candidates(query, top_k=top_k, candidate_multiplier=3)
                    
                    if use_reranking and candidates:
                        contexts = retriever.rerank_results(query, candidates, top_k=top_k)
                    else:
                        contexts = [
                            RerankedResult(
                                parent_text=r.parent_text,
                                parent_id=r.parent_id,
                                retrieval_score=r.score,
                                rerank_score=0.0,
                                child_matched=r.child_matched,
                                source_file=r.source_file
                            )
                            for r in candidates[:top_k]
                        ]
                
                # Stream the response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in generate_response_stream(query, contexts, st.session_state.rag_state.gemini_model):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                response = full_response
                
                # Show sources at the bottom
                if contexts:
                    with st.expander("📚 Sources"):
                        for i, ctx in enumerate(contexts, 1):
                            score_text = f"Rerank: {ctx.rerank_score:.3f}" if use_reranking else f"Score: {ctx.retrieval_score:.3f}"
                            st.markdown(f"**[{i}] {ctx.source_file}** ({score_text})")
                            preview = ctx.parent_text[:400] + "..." if len(ctx.parent_text) > 400 else ctx.parent_text
                            st.text(preview)
                            if i < len(contexts):
                                st.markdown("---")
                    
                    # Store sources with message
                    sources = [{"file": ctx.source_file, "text": ctx.parent_text} for ctx in contexts]
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
