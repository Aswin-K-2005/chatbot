import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)
lr = 0.01
epochs = 5000

for epoch in range(epochs):
    # Forward
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = softmax(z2)

    # One-hot encoding
    y_onehot = np.zeros((len(y_train), output_size))
    y_onehot[np.arange(len(y_train)), y_train] = 1

    # Backprop
    dz2 = y_hat - y_onehot
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 500 == 0:
        loss = -np.sum(y_onehot * np.log(y_hat + 1e-9))
        print(f"Epoch {epoch}, Loss {loss:.4f}")
import random

def predict(sentence):
    tokens = tokenize(sentence)
    bag = bag_of_words(tokens, all_words)

    z1 = np.dot(bag, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    probs = softmax(z2)

    max_prob = np.max(probs)
    predicted_index = np.argmax(probs)

    if max_prob < 0.6:   # confidence threshold
        return "unknown"

    return tags[predicted_index]


def chat():
    print("Bot is ready! Type 'quit' to exit.")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        tag = predict(sentence)
        for intent in intents['intents']:
            if intent["tag"] == tag:
                print("Bot:", random.choice(intent['responses']))
chat()
