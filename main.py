import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os
import tempfile
import shutil
import re

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

# Define model names
MODEL_NAME = "facebook/opt-350m"  # CPU-friendly, high-quality text generator
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

@st.cache_resource
def load_llm():
    """Load the language model and create a text generation pipeline."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            top_k=50,
            device=-1  # Ensure CPU usage
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def process_documents(uploaded_files):
    """Process uploaded PDF or text files into document chunks."""
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
            for i, doc in enumerate(docs):
                doc.metadata['page'] = i + 1
            documents.extend(docs)
            os.unlink(tmp_file_path)
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_rag_chain(vector_store):
    """Create a Retrieval-Augmented Generation (RAG) chain."""
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.3}
        )
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Answer the question in 1 sentence based on the context. For author queries, extract the author's name from title pages, bios, or similar sections.\nContext: {context}\nQuestion: {question}\nAnswer: "
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
st.title("Document Q&A")
st.write("Upload PDF or text files and ask questions about their content.")

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
                st.success("Documents processed successfully!")
            else:
                st.session_state.processing_complete = False
                st.error("Failed to create question-answering system.")
        else:
            st.session_state.processing_complete = False
            st.error("No valid documents processed.")
elif not uploaded_files:
    st.session_state.processing_complete = False
    st.session_state.last_uploaded_files = None
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.info("Please upload at least one document.")

# Query input
query = st.text_input(
    "Ask a question about the documents (e.g., 'Who is the author?' or 'What is the main topic?'):",
    value="name of the author",
    disabled=not st.session_state.processing_complete
)
if st.button("Get Answer", disabled=not st.session_state.processing_complete):
    with st.spinner("Generating answer..."):
        try:
            # Filter for author queries
            if "author" in query.lower():
                docs_with_scores = st.session_state.vector_store.similarity_search_with_score(
                    query, k=10, filter=lambda x: x.get('page', float('inf')) <= 5 and any(keyword in x.get('page_content', '').lower() for keyword in ['by', 'author', 'affiliation', 'email'])
                )
            else:
                docs_with_scores = st.session_state.vector_store.similarity_search_with_score(query, k=10)
            
            result = st.session_state.qa_chain.invoke({"query": query})
            answer = result["result"].strip()
            
            # Dynamic name extraction for author queries
            if "author" in query.lower():
                name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b')
                for doc in result["source_documents"]:
                    if any(keyword in doc.page_content.lower() for keyword in ['by', 'author', 'affiliation', 'email']):
                        match = name_pattern.search(doc.page_content)
                        if match and not any(cited in doc.page_content.lower() for cited in ['cite', 'reference', 'et al']):
                            answer = f"The author of the document is {match.group(1)}."
                            break
                else:
                    answer = "No author name found in the document."
            
            st.write("**Answer:**")
            st.write(answer)
            
            # Debug info for developers
            with st.expander("Debug Information (For Developers)"):
                if "author" in query.lower():
                    st.write("**Filtered Early Pages:**")
                    for doc, score in docs_with_scores[:5]:
                        st.write(f"- Page: {doc.metadata.get('page', 'N/A')}, Score: {score:.4f}, Content: {doc.page_content[:150]}...")
                st.write("**Source Documents:**")
                for doc in result["source_documents"]:
                    st.write(f"- Page: {doc.metadata.get('page', 'N/A')}, Content: {doc.page_content[:150]}...")
                st.write("**Similarity Scores:**")
                for doc, score in docs_with_scores[:5]:
                    st.write(f"- Page: {doc.metadata.get('page', 'N/A')}, Score: {score:.4f}, Content: {doc.page_content[:150]}...")
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")