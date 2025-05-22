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
import shutil  # For clearing temporary directory


# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Set device to CPU
device = "cpu"
st.write(f"Using device: {device}")

# Define model names
MODEL_NAME = "distilgpt2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-distilroberta-v1"  # Updated embedding model

@st.cache_resource
def load_llm():
    """
    Load the language model and create a text generation pipeline.
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

def create_vector_store(chunks):
    """
    Create a FAISS vector store from document chunks using embeddings.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def process_documents(uploaded_files):
    """
    Process uploaded PDF or text files into document chunks.
    """
    if not uploaded_files:
        st.error("No files uploaded.")
        return []
    
    # Clear temporary directory
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.endswith(('.pdf', '.txt')):
            os.unlink(os.path.join(temp_dir, file))
    
    documents = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1], dir=temp_dir) as tmp_file:
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
            
            docs = loader.load()
            # Add page number to metadata
            for i, doc in enumerate(docs):
                doc.metadata['page'] = i + 1
            documents.extend(docs)
            os.unlink(tmp_file_path)
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,  # Increased for better context
        chunk_overlap=150,  # Increased for continuity
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def create_rag_chain(vector_store):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.
    """
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.6}  # Adjusted for new embedding model
        )
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Use the following context to answer the question concisely in 1-2 sentences. For author-related questions, extract the author's name from title pages, copyright notices, or similar sections.\nContext: {context}\nQuestion: {question}\nAnswer: "
        )
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

# Process files only if new files are uploaded
if uploaded_files and uploaded_files != st.session_state.last_uploaded_files:
    with st.spinner("Processing documents..."):
        chunks = process_documents(uploaded_files)
        if chunks:
            vector_store = create_vector_store(chunks)
            qa_chain = create_rag_chain(vector_store)
            if qa_chain:
                st.session_state.qa_chain = qa_chain
                st.session_state.vector_store = vector_store
                st.session_state.processing_complete = True
                st.session_state.last_uploaded_files = uploaded_files
                st.success("Documents processed and RAG chain created!")
            else:
                st.session_state.processing_complete = False
                st.error("Failed to create RAG chain.")
        else:
            st.session_state.processing_complete = False
            st.error("No valid documents processed.")
elif not uploaded_files:
    st.session_state.processing_complete = False
    st.session_state.last_uploaded_files = None
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.info("Please upload at least one document to proceed.")

# Query input
query = st.text_input(
    "Ask a specific question about the documents (e.g., 'What is the main topic?' or 'Who is the author?'):",
    value="name of the author",
    disabled=not st.session_state.processing_complete
)
if st.button("Get Answer", disabled=not st.session_state.processing_complete):
    with st.spinner("Generating answer..."):
        try:
            # Prioritize early pages for author queries
            if "author" in query.lower():
                docs_with_scores = st.session_state.vector_store.similarity_search_with_score(
                    query, k=10, filter=lambda x: x.get('page', float('inf')) <= 5
                )
                st.write("**Filtered Early Pages (Debug):**")
                for doc, score in docs_with_scores:
                    st.write(f"- Page: {doc.metadata.get('page', 'N/A')}, Score: {score:.4f}, Content: {doc.page_content[:200]}...")
            else:
                docs_with_scores = st.session_state.vector_store.similarity_search_with_score(query, k=10)
            
            result = st.session_state.qa_chain.invoke({"query": query})
            st.write("**Answer:**")
            st.write(result["result"].strip())
            st.write("**Source Documents:**")
            for doc in result["source_documents"]:
                st.write(f"- Page: {doc.metadata.get('page', 'N/A')}, Content: {doc.page_content[:200]}...")
            st.write("**Retrieved Context (Debug):**")
            for doc in result["source_documents"]:
                st.write(f"- Page: {doc.metadata.get('page', 'N/A')}, Content: {doc.page_content}")
            st.write("**Similarity Scores (Debug):**")
            for doc, score in docs_with_scores:
                st.write(f"- Page: {doc.metadata.get('page', 'N/A')}, Score: {score:.4f}, Content: {doc.page_content[:200]}...")
            # Fallback: Search for Oliver Theobald
            st.write("**Fallback Check (Debug):**")
            found_author = False
            for doc in st.session_state.vector_store.similarity_search(query, k=100):
                if "Oliver Theobald" in doc.page_content or "by Oliver" in doc.page_content.lower():
                    st.write(f"- Found author: {doc.page_content[:200]}...")
                    found_author = True
                    break
            if not found_author:
                st.write("- No chunks found containing 'Oliver Theobald'.")
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")