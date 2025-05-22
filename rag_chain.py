# rag_chain.py
import streamlit as st
from google import genai
from google.genai import types
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ScientificFigureRAG:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        
        # Initialize Google GenAI client
        try:
            gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
            
            if not gemini_api_key:
                st.error("GEMINI_API_KEY not found. Please set it in Streamlit secrets.")
                logger.error("GEMINI_API_KEY not found.")
                st.stop()
                
            self.client = genai.Client(api_key=gemini_api_key)
            logger.info("Google GenAI client initialized.")
            
        except Exception as e:
            logger.error(f"Error initializing Google GenAI: {str(e)}")
            st.error(f"Error initializing Google GenAI: {str(e)}")
            st.stop()
    
    def query_figures(self, query):
        """Query the RAG system using Google GenAI"""
        try:
            # Get similar figures
            similar_figures = self.embedding_manager.search_similar_figures(query, k=5)
            
            # Format context from retrieved figures
            context = ""
            for i, (doc, score) in enumerate(similar_figures):
                context += f"\nFigure {i+1} (ID: {doc.metadata['figure_id']}, Relevance: {(1-score)*100:.1f}%):\n"
                context += f"Caption: {doc.page_content.split('Caption: ')[1].split('Domain:')[0].strip()}\n"
                context += f"Domain: {doc.metadata['domain']}\n"
                context += f"Type: {doc.metadata['figure_type']}\n"
                context += f"Keywords: {', '.join(doc.metadata['keywords'])}\n"
                context += "---\n"
            
            # Create prompt for Gemma
            prompt = f"""
You are a scientific research assistant helping researchers find relevant figures and visualizations.

Based on the following retrieved scientific figures:
{context}

User Query: {query}

Provide a structured response with:

**🎯 Most Relevant Figures:**
- List the top 3 figures with their IDs and explain why they match the query

**🔍 Analysis:**
- Identify common themes or patterns across the retrieved figures
- Explain the relevance to the user's research interest

**💡 Research Applications:**
- Suggest specific ways these figures could be useful for research
- Mention potential research directions or applications

Keep the response concise, scientific, and research-focused. Use markdown formatting for better readability.
"""

            # Generate response using Gemma
            config = types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.3
            )
            
            response = self.client.models.generate_content_stream(
                model="gemma-3-27b-it",
                contents=[prompt],
                config=config
            )
            
            # Collect the streamed response
            ai_response = "".join(chunk.text for chunk in response if chunk.text)
            logger.info(f"AI response generated: {ai_response[:100]}...")
            
            return ai_response, similar_figures
            
        except Exception as e:
            logger.error(f"Error in query_figures: {str(e)}")
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while processing your query.", []