# Import necessary libraries
import os
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# Set device (CUDA if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define paths for models
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load TinyLlama model and tokenizer
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # Create LangChain wrapper for HuggingFace pipeline
    hf_pipe = HuggingFacePipeline(pipeline=pipe)
    return hf_pipe

# Function to load and process documents
def process_documents(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue
        
        documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

# Function to create vector store from document chunks
def create_vector_store(chunks):
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    
    # Create vector store from documents
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vector_store

# Function to create RAG chain
def create_rag_chain(vector_store):
    llm = load_llm()
    
    # Create retrieval chain
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

# Add this information to the app
def about_rag():
    st.sidebar.header("About RAG")
    with st.sidebar.expander("What is RAG?"):
        st.write("""
        **Retrieval-Augmented Generation (RAG)** combines retrieval systems with
        text generation models. It works by:
        
        1. **Retrieving** relevant information from a knowledge base
        2. **Augmenting** the prompt to the LLM with this information
        3. **Generating** a response based on both the question and retrieved context
        
        This helps overcome LLM limitations by providing up-to-date and specific information.
        """)
    
    with st.sidebar.expander("Why use embeddings?"):
        st.write("""
        **Embeddings** are numerical representations of text that capture semantic meaning.
        
        Without embeddings, we would need to use basic keyword matching which:
        - Misses semantic similarities
        - Cannot understand context
        - Relies only on exact word matches
        
        Embeddings allow us to find documents that are conceptually similar to the query,
        even if they use different words.
        """)
    
    with st.sidebar.expander("How retrieval works"):
        st.write("""
        When you ask a question:
        
        1. Your question is converted to an embedding vector
        2. This vector is compared to document chunk vectors using similarity metrics
        3. The most similar chunks are retrieved
        4. These chunks provide context for the LLM's response
        
        The retriever acts as the system's "memory" while the LLM acts as the "reasoning engine".
        """)

# Streamlit interface
def main():
    st.title("📚 RAG Chatbot with TinyLlama and LangChain")
    
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    about_rag()
    
    # Process uploaded files
    if uploaded_files:
        with st.sidebar:
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily
                temp_file_paths = []
                for file in uploaded_files:
                    file_path = f"temp_{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    temp_file_paths.append(file_path)
                
                # Process documents
                chunks = process_documents(temp_file_paths)
                
                # Create vector store
                vector_store = create_vector_store(chunks)
                
                # Create QA chain
                st.session_state.qa_chain = create_rag_chain(vector_store)
                
                st.success(f"Processed {len(chunks)} document chunks!")
                
                # Clean up temporary files
                for file_path in temp_file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Check if QA chain exists
        if "qa_chain" not in st.session_state:
            with st.chat_message("assistant"):
                st.write("Please upload documents first!")
                st.session_state.messages.append({"role": "assistant", "content": "Please upload documents first!"})
        else:
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.qa_chain({"query": prompt})
                    answer = response["result"]
                    sources = response["source_documents"]
                    
                    st.write(answer)
                    
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources):
                            st.write(f"Source {i+1}:")
                            st.write(source.page_content)
                            st.write("---")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()