import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (first time only)
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function (IMPORTANT: must match training preprocessing)
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


# Load vectorizer and model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# Streamlit UI
st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check whether it is Spam or Not")

input_sms = st.text_area("Enter your message")

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Show result
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")