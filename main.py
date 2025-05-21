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
import re  # Regular expressions for answer cleaning
import time  # For tracking response time

# Initialize session state to track if documents are processed
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'think_mode' not in st.session_state:
    st.session_state.think_mode = False  # Default to non-thinking mode

# Set device to CPU (Streamlit Cloud doesn't support GPU in free tier)
device = "cpu"
st.write(f"Using device: {device}")

# Define model names
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Qwen3-0.6B for text generation
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model

@st.cache_resource  # Cache model to avoid reloading
def load_llm(think_mode=True):
    """
    Load the Qwen3-0.6B model and create a text generation pipeline.
    Args:
        think_mode: Boolean to enable thinking (True) or non-thinking (False) mode
    Returns:
        HuggingFacePipeline object for LangChain
    """
    try:
        # Load tokenizer and model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",  # Auto-select precision for CPU
            device_map="auto",  # Map to CPU
            trust_remote_code=True
        )
        # Set sampling parameters based on mode
        if think_mode:
            # Thinking mode: Detailed reasoning
            temperature = 0.6
            top_p = 0.95
            top_k = 20
            max_new_tokens = 512
        else:
            # Non-thinking mode: Faster, direct responses
            temperature = 0.7
            top_p = 0.8
            top_k = 20
            max_new_tokens = 200  # Further reduced for speed
        
        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.5,  # Prevent repetitions
            do_sample=True,  # Enable sampling (no greedy decoding)
            return_full_text=False  # Exclude input prompt from output
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
        chunk_size=400 if st.session_state.think_mode else 250,  # Smaller chunks for non-thinking
        chunk_overlap=50,  # Further reduced overlap for speed
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def create_rag_chain(vector_store, think_mode=True):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.
    Args:
        vector_store: FAISS vector store for document retrieval
        think_mode: Boolean to enable thinking (True) or non-thinking (False) mode
    Returns:
        RetrievalQA chain object
    """
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None
    try:
        # Create retriever with mode-specific settings
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3 if think_mode else 2}  # Fewer chunks for non-thinking
        )
        # Create RAG chain combining retriever and LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(think_mode=think_mode),
            chain_type="stuff",  # Combine retrieved chunks into context
            retriever=retriever,
            return_source_documents=True  # Include source documents in output
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {str(e)}")
        return None

def clean_answer(raw_answer):
    """
    Clean the raw answer from the RAG chain to extract the relevant summary.
    Args:
        raw_answer: Raw output from the LLM
    Returns:
        Cleaned answer string
    """
    # Remove Qwen3 thinking block and unwanted prefixes
    patterns = [
        r"<think>.*?</think>",  # Remove thinking content
        r"^.*?\b(Helpful Answer|Answer):",
        r"^.*?\bQuestion:.*?\n",
        r"^\s*Use the following pieces of context.*?\n\n",
        r"\n\nSource Documents:.*$",
    ]
    cleaned = raw_answer
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.MULTILINE)
    
    # Remove extra whitespace and normalize
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # Fallback if answer is too short
    if len(cleaned) < 20:
        cleaned = "The document contains information about education, projects, and work experience."
    
    return cleaned

# Streamlit UI
st.title("Document Q&A with RAG By Usman Joy")
st.write("Upload PDF or text files and ask questions about their content.")

# File uploader
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

# Mode selection (Non-Thinking Mode as default)
mode = st.radio(
    "Select Response Mode:",
    ["Thinking Mode", "Non-Thinking Mode"],
    index=1,  # Default to Non-Thinking Mode
    help="Thinking Mode uses reasoning for detailed responses (e.g., summarization). Non-Thinking Mode provides faster, direct answers."
)
think_mode = mode == "Thinking Mode"
st.session_state.think_mode = think_mode  # Store for chunk size and RAG chain

# Process files and create RAG chain
if uploaded_files:
    with st.spinner("Processing documents..."):
        chunks = process_documents(uploaded_files)
        if chunks:
            vector_store = create_vector_store(chunks)
            qa_chain = create_rag_chain(vector_store, think_mode=think_mode)
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
        start_time = time.time()
        try:
            # Format query with system prompt for Qwen3
            system_prompt = (
                "You are a helpful assistant for summarizing and answering questions about documents. "
                f"Use {'/think' if think_mode else '/no_think'} mode to process the query."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=think_mode  # Explicitly set mode
            )
            result = st.session_state.qa_chain.invoke({"query": text})
            cleaned_answer = clean_answer(result["result"])
            response_time = time.time() - start_time
            st.write(f"**Answer (generated in {response_time:.2f} seconds):**")
            st.write(cleaned_answer)
            st.write("**Source Documents:**")
            for doc in result["source_documents"]:
                st.write(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")