import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import io
import os

# Page config
st.set_page_config(page_title="SimpleRAG", page_icon="🤖", layout="wide")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize embedding model (cached)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Helper Functions
def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF safely"""
    try:
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()

        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError("File is not a valid PDF")

        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_stream)

        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()

    except Exception as e:
        return f"[Error extracting text: {str(e)}]"


def chunk_text(text, chunk_size=500, overlap=50, mode="word"):
    """Split text into overlapping chunks with different strategies"""
    
    if mode == "sentence":
        # Sentence-aware chunking
        import re
        sentences = re.split(r'[.!?]+\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_sentences = ' '.join(current_chunk).split()[-overlap:]
                current_chunk = [' '.join(overlap_sentences), sentence]
                current_length = len(overlap_sentences) + len(words)
            else:
                current_chunk.append(sentence)
                current_length += len(words)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    else:  # word mode (default)
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks


def create_vector_store(documents):
    """Create FAISS index from documents"""
    if not documents:
        return None, None

    texts = []
    for doc in documents:
        t = doc.get("text")
        if isinstance(t, str):
            texts.append(t)
        elif isinstance(t, (list, tuple)):
            texts.append(" ".join(map(str, t)))
        elif t is not None:
            texts.append(str(t))

    if not texts:
        raise ValueError("No valid text found in documents")

    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    return index, embeddings


def search_documents(query, top_k=3, similarity_threshold=2.0):
    """Search for relevant document chunks with threshold filtering"""
    if st.session_state.index is None or len(st.session_state.documents) == 0:
        return []
    
    query_embedding = embedding_model.encode([query])
    
    distances, indices = st.session_state.index.search(
        query_embedding.astype('float32'), 
        min(top_k, len(st.session_state.documents))
    )
    
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if distance <= similarity_threshold:
            results.append({
                'text': st.session_state.documents[idx]['text'],
                'source': st.session_state.documents[idx]['source'],
                'score': float(distance)
            })
    
    return results


def generate_answer(query, context_docs, response_mode, require_citations):
    """Generate answer using Gemini with enforced source consultation"""
    
    if not context_docs:
        return "I don't have enough information in the uploaded documents to answer this question."
    
    # Build numbered context with clear source markers
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(f"[SOURCE {i}] {doc['source']}\n{doc['text']}")
    
    context = "\n\n" + "\n\n".join(context_parts) + "\n\n"
    
    # Create prompt based on mode
    if response_mode == "Strict (context only)":
        base_instruction = """You are a precise assistant. Answer STRICTLY based on the provided sources.
Do NOT use any external knowledge. If the sources don't contain the answer, clearly state that."""
        
    elif response_mode == "Balanced (context + knowledge)":
        base_instruction = """You are a highly knowledgeable assistant with an IQ of 160. Use the provided sources as your PRIMARY information.
You may supplement with general knowledge ONLY if the sources are insufficient, but clearly distinguish between source information and general knowledge."""
        
    else:  # Creative
        base_instruction = """You are a creative and helpful assistant. Use the provided sources as a foundation and supplement with your knowledge to provide a comprehensive answer."""
    
    # Add citation requirements
    if require_citations:
        citation_instruction = """

CRITICAL REQUIREMENTS:
1. You MUST review ALL sources provided below
2. For EACH relevant source, include a citation like [SOURCE 1] or [SOURCE 2]
3. At the end of your answer, include a "Sources Used" section listing which sources you referenced
4. If a source was not relevant, mention it briefly in the "Sources Used" section as "not directly relevant"

This ensures accountability and shows you've consulted all available information."""
    else:
        citation_instruction = "\n\nProvide a clear and direct answer."
    
    prompt = f"""{base_instruction}{citation_instruction}

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


# UI Layout
st.title("🤖 SimpleRAG")
st.markdown("Upload documents and ask questions regarding their content! ")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    custom_key = st.toggle("Use custom Gemini keys", value=True)
    if custom_key:
        api_key = st.text_input("Enter your API key", type="password")
        if api_key:
            genai.configure(api_key=api_key)
            st.success("Custom API Key configured!")
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            st.success("API Key configured!")
    
    if not api_key:
        st.warning("Please enter your Gemini API key")
    
    st.divider()
    
    # RAG Behavior Settings
    st.header("🔍 RAG Settings")
    
    # Retrieval settings
    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="How many document chunks to retrieve for each query"
    )
    
    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="Lower = stricter matching. Only sources below this distance will be used."
    )
    
    # Chunking settings
    st.subheader("📄 Document Processing")
    chunk_mode = st.selectbox(
        "Chunking strategy",
        ["word", "sentence"],
        help="Word: splits by word count. Sentence: preserves sentence boundaries."
    )
    
    chunk_size = st.number_input(
        "Chunk size (words)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50
    )
    
    overlap = st.number_input(
        "Chunk overlap (words)",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Overlap between chunks to preserve context"
    )
    
    st.divider()
    
    # Response mode settings
    st.header("💬 Response Settings")
    
    response_mode = st.radio(
        "Response mode",
        ["Strict (context only)", "Balanced (context + knowledge)", "Creative (flexible)"],
        help="Controls how the AI uses the retrieved sources"
    )
    
    require_citations = st.checkbox(
        "Enforce source citations",
        value=True,
        help="Forces the model to cite sources and confirm it reviewed all documents"
    )
    
    require_all_sources = st.checkbox(
        "Require ALL sources consultation",
        value=True,
        help="Forces the model to mention each source, even if not directly relevant"
    )
    
    st.divider()
    
    # Document upload
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                st.session_state.documents = []
                
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    else:
                        text = uploaded_file.read().decode('utf-8')
                    
                    # Use selected chunking strategy
                    chunks = chunk_text(text, chunk_size, overlap, chunk_mode)
                    
                    for i, chunk in enumerate(chunks):
                        st.session_state.documents.append({
                            'text': chunk,
                            'source': f"{uploaded_file.name} (chunk {i+1})"
                        })
                
                st.session_state.index, st.session_state.embeddings = create_vector_store(
                    st.session_state.documents
                )
                
                st.success(f"✅ Processed {len(uploaded_files)} files into {len(st.session_state.documents)} chunks!")
    
    st.divider()
    
    # Knowledge base info
    st.header("📊 Knowledge Base")
    st.metric("Total Chunks", len(st.session_state.documents))
    
    if st.button("Clear Knowledge Base"):
        st.session_state.documents = []
        st.session_state.index = None
        st.session_state.embeddings = None
        st.session_state.chat_history = []
        st.rerun()

# Main area - Chat interface
if len(st.session_state.documents) == 0:
    st.info("👈 Upload some documents in the sidebar to get started!")
else:
    # Display current settings
    with st.expander("📋 Current RAG Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Retrieval:** Top {top_k} sources")
            st.write(f"**Threshold:** {similarity_threshold}")
            st.write(f"**Chunking:** {chunk_mode} ({chunk_size} words, {overlap} overlap)")
        with col2:
            st.write(f"**Mode:** {response_mode}")
            st.write(f"**Citations:** {'Required' if require_citations else 'Optional'}")
            st.write(f"**All sources:** {'Must consult all' if require_all_sources else 'As needed'}")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Retrieved Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['source']}")
                        st.markdown(f"**Similarity Score:** {source['score']:.3f}")
                        st.markdown(f"_{source['text'][:300]}..._")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar!")
        else:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Search documents
            with st.spinner("Searching documents..."):
                relevant_docs = search_documents(prompt, top_k, similarity_threshold)
            
            # Show warning if no sources found
            if not relevant_docs:
                st.warning(f"No sources found within similarity threshold {similarity_threshold}. Try increasing the threshold.")
            
            # Generate answer
            with st.spinner("Generating answer..."):
                answer = generate_answer(
                    prompt, 
                    relevant_docs, 
                    response_mode,
                    require_citations or require_all_sources
                )
            
            # Add assistant message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": relevant_docs
            })
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("📚 View Retrieved Sources"):
                    for i, source in enumerate(relevant_docs, 1):
                        st.markdown(f"**Source {i}:** {source['source']}")
                        st.markdown(f"**Similarity Score:** {source['score']:.3f}")
                        st.markdown(f"_{source['text'][:300]}..._")
                        st.divider()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Streamlit + Gemini API + Sentence Transformers + FAISS
</div>
""", unsafe_allow_html=True)
