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
if 'last_uploaded_image' not in st.session_state:
    st.session_state.last_uploaded_image = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Set device to CPU
device = "cpu"

# Define model names
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
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,  # Increased to allow longer responses
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            top_k=50,
            device=-1
        )
        logger.info("Language model loaded successfully.")
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Error loading language model: {str(e)}")
        st.error(f"Error loading language model: {str(e)}")
        return None

def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks."""
    logger.info("Creating vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector store created successfully.")
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
    if vector_store is None:
        logger.error("Vector store not initialized.")
        st.error("Vector store not initialized.")
        return None
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.3}
        )
        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="Generate a concise summary (1-2 sentences) based on the following image description:\nContext: {context}\nSummary: "
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        logger.info("RAG chain created successfully.")
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
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
                return
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
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            st.error(f"Error generating summary: {str(e)}")