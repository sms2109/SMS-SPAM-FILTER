import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download only stopwords (safe on cloud)
nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ==========================
# Text Preprocessing
# ==========================
def transform_text(text):
    text = text.lower()
    words = text.split()   # 🔥 replaced word_tokenize

    y = []
    for word in words:
        word = word.strip(string.punctuation)
        if word.isalnum() and word not in stop_words:
            y.append(ps.stem(word))

    return " ".join(y)

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return vectorizer, model

vectorizer, model = load_model()

# ==========================
# UI
# ==========================
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check whether it is Spam or Not.")

input_sms = st.text_area("✉️ Enter your message")

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]

        spam_prob = round(probability[1] * 100, 2)
        ham_prob = round(probability[0] * 100, 2)

        if result == 1:
            st.error("🚨 Spam Message")
            st.write(f"Spam Confidence: **{spam_prob}%**")
        else:
            st.success("✅ Not Spam")
            st.write(f"Safe Message Confidence: **{ham_prob}%**")