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
    """Extract text from a PDF safely, even if stream is unseekable or malformed."""
    try:
        # Ensure the file pointer is at start
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()

        # Verify PDF header
        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError("File is not a valid PDF")

        # Load into BytesIO for Pypdf2
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_stream)

        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()

    except Exception as e:
        return f"[Error extracting text: {str(e)}]"




def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
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

    # Clean and validate text 
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

def search_documents(query, top_k=3):
    """Search for relevant document chunks"""
    if st.session_state.index is None or len(st.session_state.documents) == 0:
        return []
    
    # Embed query
    query_embedding = embedding_model.encode([query])
    
    # Search
    distances, indices = st.session_state.index.search(
        query_embedding.astype('float32'), 
        min(top_k, len(st.session_state.documents))
    )
    
    # Return results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        results.append({
            'text': st.session_state.documents[idx]['text'],
            'source': st.session_state.documents[idx]['source'],
            'score': float(distance)
        })
    
    return results

def generate_answer(query, context_docs):
    """Generate answer using Gemini"""
    # Build context
    context = "\n\n".join([f"Source: {doc['source']}\n{doc['text']}" for doc in context_docs])
    
    # Create prompt
    prompt = f"""You are a 140 IQ highly intelligent and helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate response
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# UI Layout
st.title("🤖 SimpleRAG")
st.markdown("Upload documents and ask questions about them!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Key configured!")
    else:
        st.warning("Please enter your Gemini API key")
    
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
                    # Extract text
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    else:
                        text = uploaded_file.read().decode('utf-8')
                    
                    # Chunk text
                    chunks = chunk_text(text)
                    
                    # Add to documents
                    for i, chunk in enumerate(chunks):
                        st.session_state.documents.append({
                            'text': chunk,
                            'source': f"{uploaded_file.name} (chunk {i+1})"
                        })
                
                # Create vector store
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
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['source']}")
                        st.markdown(f"_{source['text'][:200]}..._")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar!")
        else:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get relevant documents
            with st.spinner("Searching documents..."):
                relevant_docs = search_documents(prompt, top_k=3)
            
            # Generate answer
            with st.spinner("Generating answer..."):
                answer = generate_answer(prompt, relevant_docs)
            
            # Add assistant message to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": relevant_docs
            })
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(relevant_docs, 1):
                        st.markdown(f"**Source {i}:** {source['source']}")
                        st.markdown(f"_{source['text'][:200]}..._")
                        st.divider()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Streamlit + Gemini API + Sentence Transformers + FAISS
</div>
""", unsafe_allow_html=True)
