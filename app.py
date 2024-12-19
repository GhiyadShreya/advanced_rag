
import streamlit as st
from langchain_community.llms import LlamaCpp

from llama_cpp import Llama

# Streamlit app title and description
st.title("Language Model Interaction App")
st.markdown("This app interacts with a pre-trained Llama-based language model.")

# Set up the model path (replace with your actual model path)
MODEL_PATH = "models/zephyr-7b-beta.Q4_K_M.gguf"

# Load the model
@st.cache_resource
def load_model(path):
    try:
        llm = Llama(model_path=path)
        st.success("Model loaded successfully!")
        return llm
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize the model
model = load_model(MODEL_PATH)

# User input
st.subheader("Input for Language Model")
user_input = st.text_area("Enter your prompt here:", "")

# Process the input through the model
if st.button("Generate Response"):
    if model and user_input.strip():
        try:
            response = model(user_input)
            st.subheader("Model Output")
            st.write(response['choices'][0]['text'])  # Displaying the response text
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("Please ensure the model is loaded and input is provided.")
