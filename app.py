import re
import pickle
from pathlib import Path

import nltk
import streamlit as st
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Same preprocessing as in the training notebook: lower, remove URLs, keep a-z + space, remove stopwords."""
    if text is None or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# Load the saved model and vectorizer (paths relative to this file so it works from any cwd)
_model_dir = Path(__file__).resolve().parent / "models"
with open(_model_dir / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(_model_dir / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App title and description
st.title("Toxic Comment Detector")
st.write("This tool predicts whether a comment is toxic or non-toxic.")

# Put input + submit inside a form
with st.form("toxicity_form", clear_on_submit=False):
    user_input = st.text_area("Enter a comment below:", max_chars=50_000)
    submitted = st.form_submit_button("Check Toxicity")

if submitted:
    user_input = (user_input or "").strip()
    if not user_input:
        st.warning("Please enter a comment.")
    else:
        cleaned_input = clean_text(user_input)
        if not cleaned_input.strip():
            st.warning("No meaningful text to analyze after cleaning (e.g. only numbers or punctuation).")
        else:
            input_vector = vectorizer.transform([cleaned_input])
            prediction = model.predict(input_vector)[0]
            confidence = model.predict_proba(input_vector).max()
            if prediction == 1:
                st.error(f"⚠️ Toxic Comment Detected (Confidence: {confidence:.2%})")
            else:
                st.success(f"✅ Comment is Non-Toxic (Confidence: {confidence:.2%})")
