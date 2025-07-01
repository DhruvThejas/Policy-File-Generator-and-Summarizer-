# Policy Document Explainer & Generator

A Streamlit-based application that simplifies insurance clauses and generates draft documents using NLP technology.

## Features

- Upload or paste insurance policy text
- Choose between two processing modes:
  - **Simplify**: Converts complex policy language into plain language
  - **Draft**: Creates a structured document based on policy clauses
- Process text using Google's Flan-T5 NLP model
- Display generated output based on selected mode
- Collect user feedback on results
- Download processed text as a file

## Tech Stack

- Python with Streamlit for web interface
- Hugging Face Transformers library
- PyTorch as the backend for model inference
- SentencePiece for tokenization

## Instructions

### Windows:

1. Create and navigate to project folder:
   
   cd YourProjectFolder
   

2. Create and activate a virtual environment:
   
   python -m venv venv
   venv\Scripts\activate


3. Install required libraries:

   pip install streamlit transformers torch sentencepiece


4.  Run the App
Use the following command to launch the tool locally in your browser:

   py -3.11 -m streamlit run policy_explainer.py

 It will open localhost:8501 in your browser by default.