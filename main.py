import streamlit as st  # Web app framework for Python
import torch  # Machine learning library for tensor operations
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Hugging Face library for LLMs
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Load PDF/text files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text into chunks
from langchain_community.vectorstores import FAISS  # Vector store for similarity search
from langchain.chains import RetrievalQA  # RAG chain for question answering
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings  # LangChain integration with Hugging Face
import os  # File system operations
import tempfile  # Temporary file handling

# Initialize session state to track if documents are processed
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Set device to CPU (Streamlit Cloud doesn't support GPU in free tier)
device = "cpu"
st.write(f"Using device: {device}")

# Define model names (distilgpt2 is lightweight for CPU)
MODEL_NAME = "distilgpt2"  # Small language model for text generation
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Small model for text embeddings

@st.cache_resource  # Cache model to avoid reloading
def load_llm():
    """
    Load the language model and create a text generation pipeline.
    Returns a HuggingFacePipeline object for LangChain.
    """
    try:
        # Load tokenizer and model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,  # Limit output length for CPU efficiency
            temperature=0.7,  # Control randomness (lower = less random)
            top_p=0.95,  # Control diversity (nucleus sampling)
            repetition_penalty=1.15,  # Avoid repetitive text
            do_sample=True,  # Enable sampling for varied outputs
            top_k=50,  # Consider top 50 tokens for sampling
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource  # Cache vector store to avoid recomputing
def create_vector_store(_chunks):
    """
    Create a FAISS vector store from document chunks using embeddings.
    Args:
        _chunks: List of document chunks (underscore to bypass Streamlit hashing)
    Returns:
        FAISS vector store object
    """
    try:
        # Load embedding model for converting text to vectors
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        # Create FAISS vector store (in-memory for Streamlit Cloud)
        vector_store = FAISS.from_documents(
            documents=_chunks,
            embedding=embeddings
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def process_documents(uploaded_files):
    """
    Process uploaded PDF or text files into document chunks.
    Args:
        uploaded_files: List of uploaded files from Streamlit
    Returns:
        List of document chunks
    """
    if not uploaded_files:
        st.error("No files uploaded.")
        return []
    
    documents = []
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Load file based on type
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                os.unlink(tmp_file_path)
                continue
            
            # Add loaded documents to list
            documents.extend(loader.load())
            os.unlink(tmp_file_path)  # Clean up temporary file
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Split documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Small chunks for CPU efficiency
        chunk_overlap=100,  # Overlap to maintain context
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def create_rag_chain(vector_store):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.
    Args:
        vector_store: FAISS vector store for document retrieval
    Returns:
        RetrievalQA chain object
    """
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None
    try:
        # Create retriever to find relevant document chunks
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Return top 3 relevant chunks
        )
        # Create RAG chain combining retriever and LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",  # Combine retrieved chunks into context
            retriever=retriever,
            return_source_documents=True  # Include source documents in output
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {str(e)}")
        return None

# Streamlit UI
st.title("Document Q&A with RAG")
st.write("Upload PDF or text files and ask questions about their content.")

# File uploader
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

# Process files and create RAG chain
if uploaded_files:
    with st.spinner("Processing documents..."):
        chunks = process_documents(uploaded_files)
        if chunks:
            vector_store = create_vector_store(chunks)
            qa_chain = create_rag_chain(vector_store)
            if qa_chain:
                st.session_state.qa_chain = qa_chain
                st.session_state.processing_complete = True
                st.success("Documents processed and RAG chain created!")
            else:
                st.session_state.processing_complete = False
                st.error("Failed to create RAG chain.")
        else:
            st.session_state.processing_complete = False
            st.error("No valid documents processed.")
else:
    st.session_state.processing_complete = False
    st.info("Please upload at least one document to proceed.")

# Query input (disabled until processing is complete)
query = st.text_input(
    "Ask a question about the documents:",
    value="Summarize the document",
    disabled=not st.session_state.processing_complete
)
if st.button("Get Answer", disabled=not st.session_state.processing_complete):
    with st.spinner("Generating answer..."):
        try:
            result = st.session_state.qa_chain({"query": query})
            st.write("**Answer:**")
            st.write(result["result"])
            st.write("**Source Documents:**")
            for doc in result["source_documents"]:
                st.write(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")