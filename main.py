import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # Added missing import
from langchain_core.documents import Document
from PIL import Image
import os
import tempfile

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'last_uploaded_image' not in st.session_state:
    st.session_state.last_uploaded_image = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Set device to CPU
device = "cpu"

# Define model names
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
LLM_MODEL_NAME = "facebook/opt-350m"
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

@st.cache_resource
def load_captioning_model():
    """Load the image captioning model."""
    try:
        return pipeline("image-to-text", model=CAPTION_MODEL, device=-1)
    except Exception as e:
        st.error(f"Error loading captioning model: {str(e)}")
        return None

@st.cache_resource
def load_llm():
    """Load the language model and create a text generation pipeline."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            top_k=50,
            device=-1
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading language model: {str(e)}")
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

def process_image(uploaded_image):
    """Process uploaded image to extract text and create document chunks."""
    if not uploaded_image:
        st.error("No image uploaded.")
        return []
    
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "uploaded_image.jpg")
    
    try:
        image = Image.open(uploaded_image)
        image.save(temp_file_path)
        
        captioning_model = load_captioning_model()
        if not captioning_model:
            return []
        
        caption_result = captioning_model(temp_file_path)
        image_description = caption_result[0]['generated_text'] if caption_result else ""
        
        os.unlink(temp_file_path)
        
        if not image_description:
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
        return chunks
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return []

def create_rag_chain(vector_store):
    """Create a Retrieval-Augmented Generation (RAG) chain for summarization."""
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.3}
        )
        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="Generate a concise summary (1-2 sentences) of the following image description:\nContext: {context}\nSummary: "
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
st.title("Image-Based Summary Generator")
st.write("Upload an image to generate a concise summary based on its content.")

# Image uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Process image only if a new image is uploaded
if uploaded_image and uploaded_image != st.session_state.last_uploaded_image:
    with st.spinner("Processing image..."):
        chunks = process_image(uploaded_image)
        if chunks:
            vector_store = create_vector_store(chunks)
            qa_chain = create_rag_chain(vector_store)
            if qa_chain:
                st.session_state.qa_chain = qa_chain
                st.session_state.vector_store = vector_store
                st.session_state.processing_complete = True
                st.session_state.last_uploaded_image = uploaded_image
                st.success("Image processed successfully!")
            else:
                st.session_state.processing_complete = False
                st.error("Failed to create summarization system.")
        else:
            st.session_state.processing_complete = False
            st.error("No valid content extracted from the image.")
elif not uploaded_image:
    st.session_state.processing_complete = False
    st.session_state.last_uploaded_image = None
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.info("Please upload an image.")

# Generate summary button
if st.button("Generate Summary", disabled=not st.session_state.processing_complete):
    with st.spinner("Generating summary..."):
        try:
            result = st.session_state.qa_chain.invoke({"query": "Summarize the image content"})
            summary = result["result"].strip()
            
            st.write("**Summary:**")
            st.write(summary)
            
            with st.expander("Debug Information (For Developers)"):
                st.write("**Image Description:**")
                for doc in result["source_documents"]:
                    st.write(f"- Content: {doc.page_content}")
                st.write("**Similarity Scores:**")
                docs_with_scores = st.session_state.vector_store.similarity_search_with_score("summary", k=3)
                for doc, score in docs_with_scores:
                    st.write(f"- Content: {doc.page_content[:150]}..., Score: {score:.4f}")
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")