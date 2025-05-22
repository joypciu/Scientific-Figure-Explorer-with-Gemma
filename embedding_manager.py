# embedding_manager.py
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class FigureEmbeddingManager:
    def __init__(self):
        # Lightweight embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
    
    @st.cache_data
    def load_sample_data(_self):
        """Load comprehensive sample dataset - cached for performance"""
        return {
            "figures_data": [
                {
                    "figure_id": "cnn_arch_001",
                    "caption": "Convolutional Neural Network architecture showing multiple conv layers, batch normalization, ReLU activation, max pooling, and fully connected layers for image classification tasks",
                    "domain": "Computer Vision",
                    "paper_title": "Deep CNN for Image Recognition",
                    "keywords": ["CNN", "deep learning", "computer vision", "neural network", "image classification"],
                    "figure_type": "architecture_diagram",
                    "image_url": "https://via.placeholder.com/400x300/4CAF50/white?text=CNN+Architecture"
                },
                {
                    "figure_id": "transformer_001",
                    "caption": "Transformer architecture with multi-head attention mechanism, positional encoding, encoder-decoder structure for natural language processing and machine translation tasks",
                    "domain": "Natural Language Processing", 
                    "paper_title": "Attention Is All You Need - Implementation Study",
                    "keywords": ["transformer", "attention", "NLP", "encoder-decoder", "self-attention"],
                    "figure_type": "architecture_diagram",
                    "image_url": "https://via.placeholder.com/400x300/2196F3/white?text=Transformer+Model"
                },
                {
                    "figure_id": "ml_pipeline_001",
                    "caption": "End-to-end machine learning pipeline workflow showing data ingestion, preprocessing, feature engineering, model training, hyperparameter tuning, validation, and deployment stages",
                    "domain": "Machine Learning Operations",
                    "paper_title": "Scalable ML Pipeline Design and Implementation",
                    "keywords": ["MLOps", "pipeline", "deployment", "workflow", "automation"],
                    "figure_type": "workflow_diagram", 
                    "image_url": "https://via.placeholder.com/400x300/FF9800/white?text=ML+Pipeline"
                },
                {
                    "figure_id": "gan_arch_001",
                    "caption": "Generative Adversarial Network architecture showing generator and discriminator networks in adversarial training setup with loss functions and training dynamics",
                    "domain": "Generative AI",
                    "paper_title": "Advanced GAN Architectures for Image Generation",
                    "keywords": ["GAN", "generative", "adversarial", "neural networks", "image generation"],
                    "figure_type": "architecture_diagram",
                    "image_url": "https://via.placeholder.com/400x300/9C27B0/white?text=GAN+Architecture"
                },
                {
                    "figure_id": "rl_framework_001", 
                    "caption": "Reinforcement learning framework diagram showing agent-environment interaction loop with states, actions, rewards, and policy optimization in Markov Decision Process",
                    "domain": "Reinforcement Learning",
                    "paper_title": "Deep Reinforcement Learning Methods and Applications",
                    "keywords": ["reinforcement learning", "agent", "environment", "reward", "policy", "MDP"],
                    "figure_type": "conceptual_diagram",
                    "image_url": "https://via.placeholder.com/400x300/F44336/white?text=RL+Framework"
                },
                {
                    "figure_id": "resnet_arch_001",
                    "caption": "ResNet architecture with residual connections, skip connections, and identity mappings for deep neural network training and gradient flow optimization",
                    "domain": "Computer Vision",
                    "paper_title": "Deep Residual Learning for Image Recognition",
                    "keywords": ["ResNet", "residual", "skip connections", "deep learning", "gradient flow"],
                    "figure_type": "architecture_diagram",
                    "image_url": "https://via.placeholder.com/400x300/607D8B/white?text=ResNet+Architecture"
                },
                {
                    "figure_id": "bert_arch_001",
                    "caption": "BERT model architecture showing bidirectional encoder representations, masked language modeling, and next sentence prediction for natural language understanding",
                    "domain": "Natural Language Processing",
                    "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "keywords": ["BERT", "bidirectional", "transformer", "language model", "pre-training"],
                    "figure_type": "architecture_diagram",
                    "image_url": "https://via.placeholder.com/400x300/795548/white?text=BERT+Model"
                },
                {
                    "figure_id": "autoencoder_001",
                    "caption": "Autoencoder architecture with encoder-decoder structure, latent space representation, and reconstruction loss for dimensionality reduction and feature learning",
                    "domain": "Unsupervised Learning",
                    "paper_title": "Variational Autoencoders for Representation Learning",
                    "keywords": ["autoencoder", "encoder-decoder", "latent space", "dimensionality reduction", "VAE"],
                    "figure_type": "architecture_diagram",
                    "image_url": "https://via.placeholder.com/400x300/009688/white?text=Autoencoder"
                }
            ]
        }
    
    def create_documents(self, figures_data):
        """Create documents from figure data"""
        documents = []
        for fig in figures_data:
            content = f"""
            Caption: {fig['caption']}
            Domain: {fig['domain']}
            Keywords: {', '.join(fig['keywords'])}
            Figure Type: {fig['figure_type']}
            Paper: {fig['paper_title']}
            """
            
            doc = Document(
                page_content=content.strip(),
                metadata=fig
            )
            documents.append(doc)
        return documents
    
    @st.cache_resource
    def initialize_vectorstore(_self):
        """Initialize vector store - cached to avoid recomputation"""
        data = _self.load_sample_data()
        documents = _self.create_documents(data['figures_data'])
        
        # Create FAISS vector store (in-memory)
        vectorstore = FAISS.from_documents(documents, _self.embeddings)
        logger.info(f"Vector store initialized with {len(documents)} documents")
        return vectorstore
    
    def search_similar_figures(self, query, k=5):
        """Search for similar figures"""
        if self.vectorstore is None:
            self.vectorstore = self.initialize_vectorstore()
        
        return self.vectorstore.similarity_search_with_score(query, k=k)