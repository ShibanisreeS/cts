import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = st.secrets['GENAI_API_KEY']

def chat_bot(prompt):
    genai.configure(api_key=api_key)  # Replace with your actual API key
    response = genai.chat(messages=[prompt])
    return response.last
