import re
import time
import joblib
import pandas as pd

from pathlib import Path

from transformers import pipeline
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
nltk.download("stopwords")
stop_words = set(stopwords.words("indonesian"))

import logging

logging.basicConfig(
    filename="reports/app.log",
    level=logging.INFO
)

# ==========================================================
# PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ==========================================================
# LOAD NAIVE BAYES
# ==========================================================

nb_model = joblib.load(
    MODEL_DIR / "absa_nb_model.pkl"
)

tfidf = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

# ==========================================================
# LOAD INDOBERT
# ==========================================================

bert = pipeline(
    "text-classification",
    model=str(MODEL_DIR / "bert"),
    tokenizer=str(MODEL_DIR / "bert"),
    device=0,
    top_k=None
)

# ==========================================================
# NORMALIZATION
# ==========================================================

norm = pd.read_csv(
    DATA_DIR / "normalization.csv"
)

norm_dict = dict(
    zip(
        norm["slang"],
        norm["formal"]
    )
)

# ==========================================================
# ASPECT DICTIONARY
# ==========================================================

aspect = pd.read_csv(
    DATA_DIR / "aspect_dictionary.csv"
)

aspect_dict = dict(
    zip(
        aspect["keyword"],
        aspect["aspect"]
    )
)

# ==========================================================
# STEMMER
# ==========================================================

factory = StemmerFactory()

stemmer = factory.create_stemmer()

# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess(text):

    text = str(text)

    text = text.lower()

    # url
    text = re.sub(r"http\S+", " ", text)

    # mention
    text = re.sub(r"@\w+", " ", text)

    # hashtag
    text = re.sub(r"#\w+", " ", text)

    # angka
    text = re.sub(r"\d+", " ", text)

    # simbol
    text = re.sub(r"[^\w\s]", " ", text)

    # spasi
    text = re.sub(r"\s+", " ", text).strip()

    words = []

    for word in text.split():

        if word not in stop_words:

             words.append(norm_dict.get(word, word))

    text = " ".join(words)

    text = stemmer.stem(text)

    return text

# ==========================================================
# ASPECT EXTRACTION
# ==========================================================

def extract_aspect(text):

    found = []

    lower = text.lower()

    for keyword, aspect_name in aspect_dict.items():

        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

        if re.search(pattern, lower):

            if aspect_name not in found:

                found.append(aspect_name)

    return found

# ==========================================================
# BIO TAGGING
# ==========================================================

def ner(text):

    tokens = text.split()

    result = []

    for token in tokens:

        if token.lower() in aspect_dict:

            label = "B-ASPECT"

        else:

            label = "O"

        result.append({

            "token": token,

            "label": label

        })

    return result

# ==========================================================
# NAIVE BAYES PREDICTION
# ==========================================================

def predict_nb(clean_text):

    vector = tfidf.transform([clean_text])

    sentiment = str(nb_model.predict(vector)[0]).capitalize()

    probability = nb_model.predict_proba(vector)[0]

    confidence = float(probability.max())

    return sentiment, confidence

# ==========================================================
# INDOBERT PREDICTION
# ==========================================================

def predict_bert(text):

    result = bert(text)[0]

    best = max(result, key=lambda x: x["score"])

    return best["label"], float(best["score"])

label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

def predict_bert(text):

    result = bert(text)[0]

    best = max(result, key=lambda x: x["score"])

    sentiment = label_mapping.get(best["label"], best["label"])

    return sentiment, float(best["score"])

# ==========================================================
# MAIN PREDICTION
# ==========================================================

def predict(text, model_name="Naive Bayes"):

    start = time.time()

    clean = preprocess(text)

    aspects = extract_aspect(text)

    entities = ner(text)

    if model_name == "IndoBERT":
        sentiment, confidence = predict_bert(clean)
    else:
        sentiment, confidence = predict_nb(clean)

    elapsed = round(time.time() - start, 4)

    return {
        "original_text": text,
        "preprocessed_text": clean,
        "aspect": aspects,
        "entities": entities,
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "prediction_time": elapsed,
        "model": model_name,
    }


if __name__ == "__main__":

    result = predict(
        "Packing bagus tetapi harga mahal",
        model_name="Naive Bayes"
    )

    from pprint import pprint

    pprint(result)