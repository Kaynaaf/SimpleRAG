NanoRAG: A tool to study papers effectively 🧠

# Introduction
NanoRAG is a specialized Retrieval-Augmented Generation (RAG) application designed to streamline the way researchers and students interact with academic papers. I built this project to study complex documents more effectively and to gain a hands-on understanding of how Large Language Models (LLMs) leverage external context to ground their responses.
By implementing a "Parent-Child" chunking strategy, NanoRAG ensures that the AI has access to the broader context of a document while maintaining high precision during the search phase.


# 🚀 Key Features
 * Parent-Child Retrieval: Decouples the text used for embedding (small chunks) from the text used for generation (large contexts). This allows for precise searching without losing the surrounding narrative.
 * Semantic Search & Reranking: Combines FAISS for high-speed vector similarity with a Cross-Encoder reranker to ensure the most relevant information is prioritized.
 * Multi-Format Ingestion: Seamlessly processes PDFs, Word documents (DOCX), PowerPoints (PPTX), and Text files using the MarkItDown library.
 * Streaming Responses: Real-time response generation using Google Gemini 2.0 Flash for a fast, interactive chat experience.
 * Source Attribution: Automatically cites the specific context and file name used to generate each part of the answer, ensuring academic integrity.
# 🛠️ Technical Stack
## Core Libraries
| Category | Library | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive web interface and state management. |
| Vector DB | FAISS | Efficient similarity search for dense vectors. |
| Embeddings | SentenceTransformers | Generating vector representations (all-MiniLM-L6-v2). |
| Reranking | Cross-Encoder | Refined scoring of retrieved documents (ms-marco-MiniLM-L-6-v2). |
| Parsing | MarkItDown | Microsoft’s tool for converting various file formats to Markdown. |
| LLM API | Google GenAI | Powering the final reasoning and synthesis via Gemini. |
Implementation Details
 * Chunking Logic: * Parent Size: 2000 characters (for LLM context).
   * Child Size: 400 characters (for embedding/matching).
   * Overlap: 50 characters to prevent context loss at boundaries.
 * Search Flow: Bi-Encoder Retrieval (FAISS) ➔ Cross-Encoder Reranking ➔ LLM Synthesis.
# 📊 System Architecture
The diagram below illustrates how data flows from an uploaded document to the final chat response.

```graph TD
    A[Upload Document] --> B[MarkItDown Conversion]
    B --> C[Parent-Child Splitter]
    
    subgraph Storage
        C --> D[(Parent Store: Full Text)]
        C --> E[(FAISS Index: Child Embeddings)]
    end
    
    F[User Query] --> G[SentenceTransformer]
    G --> H[FAISS Search]
    H --> I[Retrieve Parent Contexts]
    I --> J[Cross-Encoder Reranker]
    J --> K[Gemini 2.0 Flash]
    D -.-> K
    K --> L[Streaming Response + Citations]
```
# 🛠️ Getting Started
 * Clone the repository.
 * Install dependencies:
   pip install streamlit faiss-cpu sentence-transformers markitdown google-generativeai

 * Run the application:
   streamlit run app.py

 * Configure API: Enter your GEMINI_API_KEY in the sidebar or set it as an environment variable.

