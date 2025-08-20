# Lung Cancer Detection Prototype

## Overview
This Streamlit application serves as a prototype for detecting lung cancer from CT scan images using a pre-trained model hosted on Hugging Face. It also includes a chat interface powered by Groq's LLM for answering questions related to the scan or predictions. The app allows users to upload CT images, receive predictions (Benign, Malignant, or Normal), and interact with an AI assistant.

**Note:** This is a prototype and not intended for clinical use. Always consult a medical professional for health-related advice.

## Features
- **Image Upload and Prediction:** Upload a CT scan image and get a prediction using a TensorFlow model.
- **AI Chat Assistant:** Chat with an AI specializing in lung cancer detection, powered by Groq's Llama model.
- **Custom UI:** Uses custom fonts, avatars, and styling for an enhanced user experience.
- **Secure API Handling:** Groq API key is managed via Streamlit secrets.

## Requirements
The application requires Python 3.8+ and the following libraries (listed in `requirements.txt`):
- streamlit
- huggingface_hub
- tensorflow
- Pillow
- numpy
- groq

## Installation
1. Clone the repository:
