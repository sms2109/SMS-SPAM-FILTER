import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==============================
# NLTK Setup (Cloud Safe)
# ==============================

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Download only if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)

# ==============================
# Text Preprocessing
# ==============================

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ==============================
# Load Model & Vectorizer
# ==============================

@st.cache_resource
def load_model():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return vectorizer, model

vectorizer, model = load_model()

# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check whether it is Spam or Not.")

input_sms = st.text_area("✉️ Enter your message")

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]

        spam_prob = round(probability[1] * 100, 2)
        ham_prob = round(probability[0] * 100, 2)

        # Display result
        if result == 1:
            st.error("🚨 Spam Message")
            st.write(f"Spam Confidence: **{spam_prob}%**")
        else:
            st.success("✅ Not Spam")
            st.write(f"Safe Message Confidence: **{ham_prob}%**")