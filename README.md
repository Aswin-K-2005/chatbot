# 🤖 AI Chatbot (Intent-Based Neural Network)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![Type](https://img.shields.io/badge/Type-Intent--Based%20Chatbot-orange)

An intelligent intent-based chatbot built using Python and a neural network model.  
The bot classifies user input into predefined intents and generates appropriate responses.

---

## 🚀 Overview

This project implements a simple Natural Language Processing (NLP) pipeline to:

- Process user input text
- Convert text into bag-of-words representation
- Predict intent using a trained neural network
- Generate contextual responses from `intents.json`

The chatbot runs locally via command line.

---

## 🧠 How It Works (Architecture)

User Input  
↓  
Text Preprocessing (Tokenization + Lowercasing)  
↓  
Bag of Words Vectorization  
↓  
Neural Network Model (Intent Classification)  
↓  
Intent Matching (intents.json)  
↓  
Bot Response  

---

## 📂 Project Structure

```
chatbot/
│
├── new.py              # Main chatbot script
├── intents.json        # Intent dataset
├── Intent.json         # Additional intent file (if used)
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- NLTK (for NLP preprocessing)
- PyTorch (if used for model)
- JSON (for structured intent storage)

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Aswin-K-2005/chatbot.git
cd chatbot
```

### 2️⃣ Create virtual environment (Recommended)

Using Conda:

```bash
conda create -n chatbot python=3.10
conda activate chatbot
```

Or using venv:

```bash
python -m venv chatbot_env
chatbot_env\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

(If no requirements file exists, manually install required libraries.)

---

## ▶️ How To Run

```bash
python new.py
```

Example:

```
You: Hi
Bot: Hello! How can I help you today?

You: What can you do?
Bot: I can answer your questions based on my training data.
```

---

## 📊 Features

- Intent classification using machine learning
- Bag-of-Words text vectorization
- JSON-based response system
- Lightweight and runs locally

---

## 🔮 Future Improvements

- Add Flask or FastAPI web interface
- Convert into REST API
- Integrate LLM (OpenAI / HuggingFace)
- Add conversation memory
- Deploy on cloud

---

## 👨‍💻 Author

Aswin Kumar  
AI Engineering Student
