"
# Chatbot (simple NN)

Small rule-based + simple neural classifier chatbot.

## Files
- [new.py](new.py) — main script (tokenization, training loop, predictor & REPL). See [`tokenize()`](new.py), [`bag_of_words()`](new.py), [`predict()`](new.py), [`chat()`](new.py).  
- [intents.json](intents.json) — training intents used by the script.  
- [Intent.json](Intent.json) — alternative intent dataset (different schema).

## Requirements
- Python 3.7+
- numpy
- nltk

Install:
```sh
pip install numpy nltk
python -c "import nltk; nltk.download('punkt')"