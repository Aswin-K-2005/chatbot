

🤖 AI Chatbot (Intent-Based Neural Network)






An intelligent intent-based chatbot built using Python and a neural network model.
The bot classifies user input into predefined intents and generates appropriate responses.

🚀 Overview

This project implements a simple Natural Language Processing (NLP) pipeline to:

Process user input text

Convert text into bag-of-words representation

Predict intent using a trained neural network

Generate contextual responses from intents.json

The chatbot runs locally via command line and demonstrates the fundamentals of:

Text preprocessing

Intent classification

Neural network inference

Structured response generation

🧠 How It Works (Architecture)
User Input
    ↓
Text Preprocessing
(Tokenization + Lowercasing)
    ↓
Bag of Words Vectorization
    ↓
Neural Network Model
(Intent Classification)
    ↓
Intent Matching (intents.json)
    ↓
Bot Response

📂 Project Structure
chatbot/
│
├── new.py              # Main chatbot script
├── intents.json        # Intent dataset
├── Intent.json         # Additional intent file (if used)
└── README.md

🛠️ Tech Stack

Python

NLTK (for NLP preprocessing)

PyTorch / Neural Network (if used)

JSON (for structured intent storage)

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/Aswin-K-2005/chatbot.git
cd chatbot

2️⃣ Create virtual environment (Recommended)

Using Conda:

conda create -n chatbot python=3.10
conda activate chatbot


Or using venv:

python -m venv chatbot_env
chatbot_env\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt


(If no requirements file exists, manually install required libraries like NLTK or torch.)

▶️ How To Run
python new.py


Then start chatting in the terminal.

Example:

You: Hi
Bot: Hello! How can I help you today?

You: What can you do?
Bot: I can answer your questions based on my training data.

📊 Key Features

Intent classification using machine learning

Bag-of-Words text vectorization

Structured JSON-based response system

Modular and extendable design

Lightweight and runs locally

🔮 Future Improvements

Add Flask/FastAPI web interface

Convert to REST API

Integrate LLM (OpenAI / Groq / HuggingFace)

Add conversation memory

Use embeddings instead of bag-of-words

Deploy on cloud (Render / Railway / AWS)

🎯 Learning Outcomes

This project demonstrates understanding of:

NLP preprocessing pipeline

Intent-based chatbot design

Neural network-based classification

Python project structuring

Environment management

📌 Author

Aswin Kumar
AI Engineering Student | Building AI Systems
