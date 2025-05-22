<<<<<<< HEAD
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from PIL import Image
import os
import tempfile
import logging
=======
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
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Workaround for Streamlit-PyTorch conflict
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'poll'

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
<<<<<<< HEAD
if 'last_uploaded_image' not in st.session_state:
    st.session_state.last_uploaded_image = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
=======
if 'think_mode' not in st.session_state:
    st.session_state.think_mode = False  # Default to non-thinking mode
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)

# Set device to CPU
device = "cpu"

# Define model names
<<<<<<< HEAD
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
LLM_MODEL_NAME = "distilgpt2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(max_entries=1)
def load_captioning_model():
    """Load the image captioning model."""
    logger.info("Loading captioning model...")
    try:
        model = pipeline("image-to-text", model=CAPTION_MODEL, device=-1, use_fast=True)
        logger.info("Captioning model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading captioning model: {str(e)}")
        st.error(f"Error loading captioning model: {str(e)}")
        return None

@st.cache_resource(max_entries=1)
def load_llm():
    """Load the language model and create a text generation pipeline."""
    logger.info("Loading language model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
=======
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
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
<<<<<<< HEAD
            max_new_tokens=100,  # Increased to allow longer responses
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            top_k=50,
            device=-1
=======
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.5,  # Prevent repetitions
            do_sample=True,  # Enable sampling (no greedy decoding)
            return_full_text=False  # Exclude input prompt from output
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)
        )
        logger.info("Language model loaded successfully.")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Error loading language model: {str(e)}")
        st.error(f"Error loading language model: {str(e)}")
        return None

<<<<<<< HEAD
def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks."""
    logger.info("Creating vector store...")
=======
@st.cache_resource  # Cache vector store to avoid recomputing
def create_vector_store(_chunks):
    """
    Create a FAISS vector store from document chunks using embeddings.
    Args:
        _chunks: List of document chunks (underscore to bypass Streamlit hashing)
    Returns:
        FAISS vector store object
    """
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)
    try:
        # Load embedding model for converting text to vectors
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
<<<<<<< HEAD
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector store created successfully.")
=======
        # Create FAISS vector store (in-memory for Streamlit Cloud)
        vector_store = FAISS.from_documents(
            documents=_chunks,
            embedding=embeddings
        )
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error(f"Error creating vector store: {str(e)}")
        return None

def process_image(uploaded_image):
    """Process uploaded image to extract text and create document chunks."""
    logger.info("Processing image...")
    if not uploaded_image:
        logger.warning("No image uploaded.")
        st.error("No image uploaded.")
        return []
    
<<<<<<< HEAD
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "uploaded_image.jpg")
    
    try:
        image = Image.open(uploaded_image).resize((512, 512))
        image.save(temp_file_path)
        logger.info(f"Image saved to {temp_file_path}")
        
        captioning_model = load_captioning_model()
        if not captioning_model:
            logger.error("Captioning model not loaded.")
            return []
        
        caption_result = captioning_model(temp_file_path)
        image_description = caption_result[0]['generated_text'] if caption_result else ""
        logger.info(f"Image description: {image_description}")
        
        os.unlink(temp_file_path)
        logger.info(f"Temporary file {temp_file_path} deleted.")
        
        if not image_description:
            logger.warning("No meaningful content extracted from the image.")
            st.warning("No meaningful content extracted from the image.")
            return []
        
        document = Document(
            page_content=image_description,
            metadata={"source": "image_description"}
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=40,
            length_function=len
        )
        chunks = text_splitter.split_documents([document])
        logger.info(f"Created {len(chunks)} document chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error(f"Error processing image: {str(e)}")
        return []

def create_rag_chain(vector_store):
    """Create a Retrieval-Augmented Generation (RAG) chain for summarization."""
    logger.info("Creating RAG chain...")
=======
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
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)
    if vector_store is None:
        logger.error("Vector store not initialized.")
        st.error("Vector store not initialized.")
        return None
    try:
        # Create retriever with mode-specific settings
        retriever = vector_store.as_retriever(
<<<<<<< HEAD
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.3}
        )
        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="Generate a concise summary (1-2 sentences) based on the following image description:\nContext: {context}\nSummary: "
        )
=======
            search_type="similarity",
            search_kwargs={"k": 3 if think_mode else 2}  # Fewer chunks for non-thinking
        )
        # Create RAG chain combining retriever and LLM
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(think_mode=think_mode),
            chain_type="stuff",  # Combine retrieved chunks into context
            retriever=retriever,
            return_source_documents=True  # Include source documents in output
        )
        logger.info("RAG chain created successfully.")
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
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
<<<<<<< HEAD
st.title("Image-Based Summary Generator")
st.write("Upload an image to generate a concise summary based on its content.")
=======
st.title("Document Q&A with RAG By Usman Joy")
st.write("Upload PDF or text files and ask questions about their content.")
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)

# Image uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

<<<<<<< HEAD
# Process image only if a new image is uploaded
if uploaded_image and uploaded_image != st.session_state.last_uploaded_image:
    with st.spinner("Processing image..."):
        logger.info("Starting image processing for new upload.")
        chunks = process_image(uploaded_image)
        if chunks:
            vector_store = create_vector_store(chunks)
            if vector_store:
                qa_chain = create_rag_chain(vector_store)
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.session_state.vector_store = vector_store
                    st.session_state.processing_complete = True
                    st.session_state.last_uploaded_image = uploaded_image
                    st.success("Image processed successfully!")
                    logger.info("Image processed and RAG chain created successfully.")
                else:
                    st.session_state.processing_complete = False
                    st.error("Failed to create summarization system.")
                    logger.error("Failed to create summarization system.")
=======
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
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)
            else:
                st.session_state.processing_complete = False
                st.error("Failed to create vector store.")
                logger.error("Failed to create vector store.")
        else:
            st.session_state.processing_complete = False
            st.error("No valid content extracted from the image.")
            logger.error("No valid content extracted from the image.")
elif not uploaded_image:
    st.session_state.processing_complete = False
    st.session_state.last_uploaded_image = None
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.info("Please upload an image.")
    logger.info("No image uploaded, resetting session state.")

<<<<<<< HEAD
# Generate summary button
if st.button("Generate Summary", disabled=not st.session_state.processing_complete):
    with st.spinner("Generating summary..."):
        logger.info("Generating summary...")
        try:
            # Ensure context is retrieved and passed correctly
            context = st.session_state.vector_store.as_retriever().get_relevant_documents("Summarize the image content")
            if not context:
                st.error("No relevant context found for summarization.")
                logger.error("No relevant context found for summarization.")
                pass  # Use 'pass' instead of 'return'
            result = st.session_state.qa_chain.invoke({"query": "Summarize the image content"})
    
            summary = result["result"].strip()
            
            st.write("**Summary:**")
            st.write(summary if summary != prompt_template.template else "Failed to generate summary.")
            logger.info(f"Summary generated: {summary}")
            
            with st.expander("Debug Information (For Developers)"):
                st.write("**Image Description:**")
                for doc in result["source_documents"]:
                    st.write(f"- Content: {doc.page_content}")
                st.write("**Similarity Scores:**")
                docs_with_scores = st.session_state.vector_store.similarity_search_with_score("summary", k=3)
                for doc, score in docs_with_scores:
                    st.write(f"- Content: {doc.page_content[:150]}..., Score: {score:.4f}")
=======
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
>>>>>>> parent of be16100 (commiting distilgpt2 with new task)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            st.error(f"Error generating summary: {str(e)}")