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
from langchain.prompts import PromptTemplate  # Import PromptTemplate

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Set device to CPU
device = "cpu"
st.write(f"Using device: {device}")

# Define model names
MODEL_NAME = "distilgpt2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_llm():
    """
    Load the language model and create a text generation pipeline optimized for question answering.
    Returns a HuggingFacePipeline object for LangChain.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            top_k=40,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def create_vector_store(_chunks):
    """
    Create a FAISS vector store from document chunks using embeddings.
    Args:
        _chunks: List of document chunks
    Returns:
        FAISS vector store object
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                os.unlink(tmp_file_path)
                continue
            
            documents.extend(loader.load())
            os.unlink(tmp_file_path)
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def create_rag_chain(vector_store):
    """
    Create a Retrieval-Augmented Generation (RAG) chain with a custom prompt for concise answers.
    Args:
        vector_store: FAISS vector store for document retrieval
    Returns:
        RetrievalQA chain object
    """
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        # Define a proper PromptTemplate object
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Use the following context to answer the question concisely in 1-2 sentences. Focus on clarity and relevance.\nContext: {context}\nQuestion: {question}\nAnswer: "
        )
        # Create RAG chain with the prompt template
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {str(e)}")
        return None

# Streamlit UI
st.title("Document Q&A with RAG")
st.write("Upload PDF or text files and ask specific questions about their content.")

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

# Query input
query = st.text_input(
    "Ask a specific question about the documents (e.g., 'What is the main topic?' or 'Who is mentioned?'):",
    value="What is the main topic of the document?",
    disabled=not st.session_state.processing_complete
)
if st.button("Get Answer", disabled=not st.session_state.processing_complete):
    with st.spinner("Generating answer..."):
        try:
            result = st.session_state.qa_chain.invoke({"query": query})
            st.write("**Answer:**")
            st.write(result["result"].strip())
            st.write("**Source Documents:**")
            for doc in result["source_documents"]:
                st.write(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")