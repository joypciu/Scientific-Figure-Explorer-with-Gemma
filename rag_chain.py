# rag_chain.py
import streamlit as st
from openai import OpenAI
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ScientificFigureRAG:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        
        # Initialize OpenRouter client
        try:
            openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            site_url = st.secrets.get("SITE_URL") or os.getenv("SITE_URL", "https://localhost:8501")
            site_name = st.secrets.get("SITE_NAME") or os.getenv("SITE_NAME", "Scientific Figure Explorer")
            
            if not openrouter_api_key:
                st.error("OPENROUTER_API_KEY not found. Please set it in Streamlit secrets.")
                logger.error("OPENROUTER_API_KEY not found.")
                st.stop()
                
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
            )
            
            # Store headers for requests
            self.extra_headers = {
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            }
            
            logger.info("OpenRouter client initialized.")
            
        except Exception as e:
            logger.error(f"Error initializing OpenRouter: {str(e)}")
            st.error(f"Error initializing OpenRouter: {str(e)}")
            st.stop()
    
    def query_figures(self, query):
        """Query the RAG system using OpenRouter's Gemma 3N model"""
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
            
            # Create prompt for Gemma 3N
            prompt = f"""You are a scientific research assistant helping researchers find relevant figures and visualizations.

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

Keep the response concise, scientific, and research-focused. Use markdown formatting for better readability."""

            # Generate response using Gemma 3N via OpenRouter
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                extra_body={},
                model="google/gemma-3n-e4b-it:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            ai_response = completion.choices[0].message.content
            logger.info(f"AI response generated: {ai_response[:100]}...")
            
            return ai_response, similar_figures
            
        except Exception as e:
            logger.error(f"Error in query_figures: {str(e)}")
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while processing your query.", []