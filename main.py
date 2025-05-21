import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import os
import tempfile

# Initialize session state
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Set device (use CPU if no GPU is available)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")

# Define model names
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cache the language model to avoid reloading
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        top_k=50,
    )
    return HuggingFacePipeline(pipeline=pipe)

# Load the model
llm = load_llm()
st.success("Language model loaded successfully!")

def process_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load the file based on its type
        try:
            if tmp_file_path.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif tmp_file_path.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                continue
            documents.extend(loader.load())
        finally:
            os.unlink(tmp_file_path)  # Delete temporary file
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

@st.cache_resource
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    # Check if FAISS index exists
    index_path = "./faiss_index"
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        st.write("Loaded existing FAISS index.")
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
        st.write("Created and saved new FAISS index.")
    return vector_store

def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Streamlit UI
st.title("Document Q&A with RAG")
st.write("Upload PDF or text files and ask questions about their content.")

# File uploader
uploaded_files = st.file_uploader("Upload PDF or Text Files", type=["pdf", "txt"], accept_multiple_files=True)

# Process uploaded files
if uploaded_files:
    with st.spinner("Processing documents..."):
        chunks = process_documents(uploaded_files)
        if chunks:
            vector_store = create_vector_store(chunks)
            st.success("Vector store created successfully!")
            st.session_state.qa_chain = create_rag_chain(vector_store)
            st.success("RAG chain created successfully!")
            st.session_state.documents_processed = True
        else:
            st.error("No valid documents processed.")
            st.session_state.documents_processed = False
            st.session_state.qa_chain = None
else:
    st.session_state.documents_processed = False
    st.session_state.qa_chain = None
    st.info("Please upload at least one PDF or text file.")

# Query input (disabled until documents are processed)
query = st.text_input(
    "Ask a question about the documents:",
    value="Summarize the document",
    disabled=not st.session_state.documents_processed
)

# Submit button (disabled until documents are processed)
if st.button("Get Answer", disabled=not st.session_state.documents_processed):
    with st.spinner("Generating answer..."):
        result = st.session_state.qa_chain({"query": query})
        st.write("**Answer:**")
        st.write(result["result"])
        st.write("**Source Documents:**")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")