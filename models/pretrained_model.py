# pretrained_model.py

import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


# ✅ Cached model loader — placed outside the class
@st.cache_resource
def get_cached_model():
    try:
        llm = HuggingFacePipeline.from_model_id(
            model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            task='text-generation',
             model_kwargs={"device_map": "cpu"},
            pipeline_kwargs={
                'temperature': 0.5,
                'max_new_tokens': 100
            }
        )
        return ChatHuggingFace(llm=llm)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


class LLMChatApp:
    def __init__(self):
        self.hf_token = None
        self.model = None

    def load_env_and_login(self):
        """Load Hugging Face token from .env and login."""
        load_dotenv()
        self.hf_token = os.getenv("HF_AUTH_TOKEN")

        if not self.hf_token:
            st.error("🔐 Hugging Face token not found. Please check your .env file.")
            return False

        try:
            login(token=self.hf_token)
            return True
        except Exception as e:
            st.error(f"❌ Hugging Face login failed: {e}")
            return False

    def load_model(self):
        """Load the cached Hugging Face model."""
        self.model = get_cached_model()  # ✅ correct global call
        if self.model:
            return self.model
        else:
            st.error("⚠️ Failed to load the model.")
            return None
