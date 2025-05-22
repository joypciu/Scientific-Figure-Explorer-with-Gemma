# app.py
import streamlit as st
from embedding_manager import FigureEmbeddingManager
from rag_chain import ScientificFigureRAG
import requests
from PIL import Image
from io import BytesIO
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Workaround for Streamlit file watcher (from your code)
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'poll'

# Page config
st.set_page_config(
    page_title="Scientific Figure Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling (enhanced version of yours)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #0f1419, #1a2332);
        color: #ffffff;
    }
    .figure-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid #00d4aa;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #00d4aa;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #00b894;
        transform: translateY(-1px);
    }
    .metric-card {
        background: rgba(0, 212, 170, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

def load_image_from_url(url):
    """Load image from URL with error handling"""
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        logger.error(f"Error loading image from {url}: {str(e)}")
        return None

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system - cached for performance"""
    try:
        embedding_manager = FigureEmbeddingManager()
        rag_system = ScientificFigureRAG(embedding_manager)
        logger.info("RAG system initialized successfully")
        return rag_system
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        raise e

def main():
    st.title("🔬 Scientific Figure Explorer")
    st.markdown("*Discover relevant scientific figures using RAG and Google GenAI*")
    
    # Sidebar info
    with st.sidebar:
        st.header("🎯 About This Tool")
        st.markdown("""
        This RAG system helps researchers find relevant scientific figures and visualizations using:
        - **Semantic Search**: Find figures by meaning, not just keywords
        - **AI Analysis**: Get insights powered by Google Gemma
        - **Multi-domain Coverage**: ML, CV, NLP, and more
        """)
        
        st.header("💡 Example Queries")
        example_queries = [
            "CNN architecture for image classification",
            "Transformer model with attention mechanism", 
            "Machine learning pipeline workflow",
            "GAN architecture for image generation",
            "Reinforcement learning framework",
            "Autoencoder for dimensionality reduction"
        ]
        
        for query in example_queries:
            if st.button(f"📝 {query}", key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.example_query = query
        
        st.header("📊 Dataset Info")
        st.info("Currently indexing 8 scientific figures across multiple domains")
    
    # Initialize system
    try:
        rag_system = initialize_rag_system()
        st.success("✅ RAG System Ready")
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        st.stop()
    
    # Main interface
    query = st.text_input(
        "🔍 Search for scientific figures:",
        value=st.session_state.get('example_query', ''),
        placeholder="Describe the type of figure you're looking for...",
        help="Enter a description of the scientific figure or concept you want to explore"
    )
    
    # Clear the example query after use
    if 'example_query' in st.session_state:
        del st.session_state.example_query
    
    if query:
        with st.spinner("🔍 Searching scientific literature..."):
            try:
                response, retrieved_figures = rag_system.query_figures(query)
                
                # Display AI analysis
                st.subheader("🤖 AI Analysis")
                st.markdown(response)
                
                st.markdown("---")
                
                # Display retrieved figures
                st.subheader("📊 Retrieved Figures")
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Figures Found", len(retrieved_figures))
                with col2:
                    avg_relevance = sum((1-score)*100 for _, score in retrieved_figures) / len(retrieved_figures)
                    st.metric("Avg Relevance", f"{avg_relevance:.1f}%")
                with col3:
                    domains = list(set(doc.metadata['domain'] for doc, _ in retrieved_figures))
                    st.metric("Domains", len(domains))
                
                # Display top figures
                for i, (doc, score) in enumerate(retrieved_figures[:4]):  # Show top 4
                    relevance_score = (1-score)*100
                    
                    with st.expander(f"📈 Figure {i+1}: {doc.metadata['figure_id']} (Relevance: {relevance_score:.1f}%)", expanded=i<2):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Load and display image
                            img = load_image_from_url(doc.metadata.get('image_url', ''))
                            if img:
                                st.image(img, use_column_width=True)
                            else:
                                st.info("📷 Image placeholder")
                        
                        with col2:
                            # Figure details
                            st.markdown(f"**📄 Paper:** {doc.metadata.get('paper_title', 'N/A')}")
                            st.markdown(f"**🏷️ Domain:** {doc.metadata.get('domain', 'N/A')}")
                            st.markdown(f"**📊 Type:** {doc.metadata.get('figure_type', 'N/A')}")
                            st.markdown(f"**🔤 Keywords:** {', '.join(doc.metadata.get('keywords', []))}")
                            
                            # Extract caption from content
                            caption = doc.page_content.split('Caption: ')[1].split('\nDomain:')[0].strip()
                            st.markdown(f"**📝 Caption:** {caption}")
                            
                            # Relevance indicator
                            if relevance_score > 80:
                                st.success(f"🎯 High relevance: {relevance_score:.1f}%")
                            elif relevance_score > 60:
                                st.info(f"✅ Good match: {relevance_score:.1f}%")
                            else:
                                st.warning(f"⚠️ Moderate match: {relevance_score:.1f}%")
                            
            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                st.error(f"❌ Search failed: {str(e)}")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔧 **Built with:**")
        st.markdown("- LangChain & FAISS")
    with col2:
        st.markdown("🤖 **Powered by:**")
        st.markdown("- Google GenAI (Gemma)")
    with col3:
        st.markdown("🚀 **Deployed on:**")
        st.markdown("- Streamlit Cloud")

if __name__ == "__main__":
    main()