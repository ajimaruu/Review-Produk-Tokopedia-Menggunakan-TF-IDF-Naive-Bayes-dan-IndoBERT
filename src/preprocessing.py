import re
import pandas as pd
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# ===========================
# Load Data
# ===========================

df = pd.read_csv("data/processed/absa_ready.csv")

normalization = pd.read_csv("data/normalization.csv")

# ===========================
# Normalization Dictionary
# ===========================

norm_dict = dict(zip(normalization.slang, normalization.formal))

# ===========================
# Stopword
# ===========================

stop_words = set(stopwords.words("indonesian"))

# ===========================
# Stemmer
# ===========================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):

    text = str(text).lower()

    # URL
    text = re.sub(r"http\S+", "", text)

    # Mention
    text = re.sub(r"@\w+", "", text)

    # Hashtag
    text = re.sub(r"#\w+", "", text)

    # Angka
    text = re.sub(r"\d+", "", text)

    # Emoji
    text = re.sub(r"[^\w\s]", " ", text)

    # Spasi ganda
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize(sentence):

    result = []

    for word in sentence.split():

        result.append(norm_dict.get(word, word))

    return " ".join(result)

def remove_stopwords(sentence):

    words = []

    for word in sentence.split():

        if word not in stop_words:

            words.append(word)

    return " ".join(words)

def stemming(sentence):

    return stemmer.stem(sentence)

before = []
after = []

for text in df["text"]:

    original = text

    text = clean_text(text)

    text = normalize(text)

    text = remove_stopwords(text)

    text = stemming(text)

    before.append(original)

    after.append(text)

df["before"] = before
df["after"] = after

df.to_csv(

    "data/processed/preprocessed.csv",

    index=False

)

print(df[["before","after"]].head())