import streamlit as st
from huggingface_hub import snapshot_download
import tensorflow as tf
from PIL import Image
import numpy as np
import groq
import os

# --- UI CONFIG & STYLE ---
st.set_page_config(page_title="DigiTwin RAG Forecast", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * {
        font-family: 'Tw Cen MT', sans-serif !important;
    }

    /* Sidebar arrow fix */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "▶";
        font-size: 1.3rem;
        margin-right: 0.4rem;
    }

    /* Top-right logo placement */
    .logo-container {
        position: fixed;
        top: 5rem;
        right: 12rem;
        z-index: 9999;
    }
    </style>
""", unsafe_allow_html=True)

# Display logo (smaller, top-right)
st.markdown(
    """
    <div class="logo-container">
        <img src="https://github.com/valonys/DigiTwin/blob/29dd50da95bec35a5abdca4bdda1967f0e5efff6/ValonyLabs_Logo.png?raw=true" width="70">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("📊 DigiTwin - The Insp Nerdzx")

# --- AVATARS ---
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# Sidebar for file upload
st.sidebar.title("Upload CT Scan Image")
uploaded_file = st.sidebar.file_uploader("Choose a CT image...", type=["jpg", "png", "jpeg"])

# Load Groq API key from secrets
if "groq_api_key" in st.secrets:
    groq_api_key = st.secrets["groq_api_key"]
else:
    groq_api_key = None
    st.sidebar.warning("Groq API key not found. Add it to .streamlit/secrets.toml as groq_api_key = 'your_key_here'")

# Load the pre-trained model from Hugging Face
@st.cache_resource
def load_model():
    model_dir = snapshot_download(repo_id="Chinwendu/lung_ct_detection_model")
    return tf.keras.models.load_model(model_dir)

model = load_model()

# Main content area
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded CT Scan Image', use_column_width=True)
    
    # Preprocess the image
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    class_names = ['Benign', 'Malignant', 'Normal']
    predicted_class = class_names[np.argmax(prediction)]
    
    st.success(f"Prediction: {predicted_class}")

# Chat UI in the main area
st.header("Chat with the AI Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else BOT_AVATAR):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the scan or prediction"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Generate a response using Groq API if API key is provided
    if groq_api_key:
        try:
            client = groq.Groq(api_key=groq_api_key)
            predicted_class = locals().get('predicted_class', 'No prediction yet')
            system_prompt = f"You are a medical AI assistant specializing in lung cancer detection. Based on the uploaded scan, the model predicts: {predicted_class}. Answer the user's question accurately and helpfully, providing explanations where necessary. Do not provide medical advice; suggest consulting a doctor."
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",  # Stable model as per Groq documentation
                temperature=0.7,
                max_tokens=1024,
            )
            
            response = chat_completion.choices[0].message.content
        except Exception as e:
            response = f"Error generating response: {str(e)}. Please check your API key and try again."
    else:
        response = "Groq API key not configured. Please add it to your secrets.toml file."
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(response)

# Add powered by text below chat input
st.markdown("Powered by MedLung Intelli Systems | Your saying | ValonyLabs <a href='https://www.valonylabs.com'>www.valonylabs.com</a>", unsafe_allow_html=True)
