import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import altair as alt

# --- DOWNLOAD NLTK DATA ---
# This must happen before you try to use word_tokenize or stopwords
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

download_nltk_data()

# --- Custom CSS for a Netflix-like dark theme ---
def set_netflix_style():
    st.markdown(
        """
        <style>
        .stApp { background-color: #141414; color: #E5E5E5; }
        h1, h2, h3 { color: #FFFFFF; text-align: center; }
        .stButton>button { background-color: #E50914; color: white; font-weight: bold; border: none; }
        .stButton>button:hover { background-color: #B20710; border: none; }
        input { background-color: #222222 !important; color: white !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Helper Functions ---
@st.cache_resource
def load_resources():
    # FIXED: Changed paths from Google Drive to local relative paths
    # Ensure these files are in your GitHub repository!
    model_path = 'toxicity_model.keras'
    tokenizer_path = 'tokenizer.pickle'

    try:
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            st.error(f"Files missing! Make sure {model_path} and {tokenizer_path} are in your GitHub repo.")
            return None, None
            
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

def preprocess_text(text, tokenizer, max_len=200):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    preprocessed_text = " ".join(filtered_tokens)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    return pad_sequences(sequence, maxlen=max_len, padding='post')

def predict_toxicity(text, model, tokenizer):
    if model is None or tokenizer is None:
        return "N/A"
    padded_sequence = preprocess_text(text, tokenizer)
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    return "Toxic" if prediction > 0.5 else "Not Toxic"

# --- Main App ---
def main():
    set_netflix_style()
    st.title("Comment Toxicity Detection 💬")
    
    model, tokenizer = load_resources()
    
    tab1, tab2 = st.tabs(["Real-Time", "Bulk CSV"])

    with tab1:
        comment = st.text_area("Enter a comment:")
        if st.button('Analyze'):
            if comment:
                result = predict_toxicity(comment, model, tokenizer)
                if result == "Toxic":
                    st.error(f"Prediction: {result}")
                else:
                    st.success(f"Prediction: {result}")

    with tab2:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file and model:
            df = pd.read_csv(uploaded_file)
            if 'comment_text' in df.columns:
                with st.spinner('Analyzing...'):
                    df['prediction'] = df['comment_text'].apply(lambda x: predict_toxicity(x, model, tokenizer))
                st.dataframe(df)
                st.download_button("Download Results", df.to_csv(index=False), "results.csv", "text/csv")
            else:
                st.error("CSV must have a 'comment_text' column.")

if __name__ == '__main__':
    main()
