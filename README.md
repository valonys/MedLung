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

The app will open in your default web browser.

## Usage
1. **Upload Image:** Use the sidebar to upload a CT scan image (JPG, PNG, JPEG).
2. **View Prediction:** The app will display the image and predict if it's Benign, Malignant, or Normal.
3. **Chat Interface:** Ask questions about the scan or prediction in the chat input. The AI assistant will respond based on the prediction.
4. **Powered By:** Footer credits to MedLung Intelli Systems, Your saying, and ValonyLabs.

## Model Details
- **Prediction Model:** Loaded from Hugging Face repo `Chinwendu/lung_ct_detection_model`.
- **Chat Model:** Uses Groq's `llama-3.1-8b-instant` model for responses.

## Limitations
- The prediction model is pre-trained and may not be 100% accurate.
- Chat responses are AI-generated and should not replace professional medical advice.
- No internet access for additional package installations in the code interpreter tool (if used in extensions).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions, contact ValonyLabs at [www.valonylabs.com](https://www.valonylabs.com).
