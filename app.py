import streamlit as st
import pandas as pd
import pickle
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack, csr_matrix
from bs4 import BeautifulSoup

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Advanced Spam Detector", page_icon="🛡️")

# Download NLTK data (Cached so it doesn't re-download on every run)
@st.cache_resource
def download_nltk_data():
    resources = ['punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
    return True

download_nltk_data()

# --- 2. LOAD SAVED ARTIFACTS ---
@st.cache_resource
def load_pipeline():
    with open('spam_classifier_pipeline.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    # --- PATCH FOR VERSION MISMATCH ---
    # If the scaler was trained on an old sklearn version, it might miss 'clip'
    scaler_obj = artifacts["scaler"]
    if not hasattr(scaler_obj, "clip"):
        scaler_obj.clip = False
    
    return artifacts

data = load_pipeline()
model = data["model"]
tfidf = data["tfidf"]
scaler = data["scaler"]
rare_words = data["rare_words"]
spam_likelihood_dict = data["spam_likelihood_dict"]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- 3. RECREATE YOUR CLEANING & FEATURE LOGIC ---

def clean_text(html_text):
    """
    Cleans raw email text by:
    1. Parsing HTML and extracting plain text.
    2. Normalizing whitespace (newlines, tabs, etc.).
    3. Removing URLs, email addresses, and all non-alphabetic characters.
    4. Lowercasing and stripping final whitespace.
    """
    # 1. Parse HTML and extract plain text
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        text = soup.get_text()
    except:
        # Handle cases where the body might not be HTML (e.g., just plain text)
        # Ensure the input is treated as a string
        text = str(html_text)

    # 2. Normalize whitespace (replace multiple spaces, newlines, tabs with a single space)
    text = re.sub(r'\s+', ' ', text)

    # 3. Remove noise: URLs, emails, and punctuation
    text = re.sub(r'http\S+', ' ', text)      # Replace URLs with a space
    text = re.sub(r'\S+@\S+', ' ', text)     # Replace emails with a space
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Replace non-letters/non-spaces with a space

    # 4. Final cleanup: lowercase and strip leading/trailing spaces
    text = text.lower().strip()
    
    return text

def preprocess_text_2(text):
    """NLTK tokenization and lemmatization"""
    tokens = word_tokenize(text)
    processed_tokens = []
    for word in tokens:
        if word not in stop_words:
            processed_tokens.append(lemmatizer.lemmatize(word))
    
    if len(processed_tokens) < 1:
        return "" # Return empty string instead of None to prevent errors
    return " ".join(processed_tokens)

# --- Feature Engineering Functions ---

def keyword_likelihood_score(text):
    score = 0
    # We use the dictionary loaded from the pickle file
    for kw, likelihood in spam_likelihood_dict.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', text, flags=re.IGNORECASE):
            score += likelihood
    return score

def count_rare_words(text):
    # We use the rare_words set loaded from the pickle file
    return sum(1 for w in str(text).split() if w in rare_words)

def extract_features(raw_text, processed_text):
    """
    Recreates the exact numeric features used in training:
    ['kw_score', 'word_count', 'unique_word_count', 'rare_word_count']
    """
    # Feature 1: Keyword Score
    kw_score = keyword_likelihood_score(processed_text)
    
    # Feature 2: Word Count
    word_count = len(str(processed_text).split())
    
    # Feature 3: Unique Word Count
    unique_word_count = len(set(str(processed_text).split()))
    
    # Feature 4: Rare Word Count
    r_word_count = count_rare_words(processed_text)
    
    # Create DataFrame with exact column order as training
    features_df = pd.DataFrame({
        'kw_score': [kw_score],
        'word_count': [word_count],
        'unique_word_count': [unique_word_count],
        'rare_word_count': [r_word_count]
    })
    
    return features_df

# --- 4. STREAMLIT UI ---

st.title("🛡️ Enterprise Email Spam Classifier")
st.markdown("""
This model uses a Hybrid approach: **TF-IDF Vectorization** + **Custom Feature Engineering** (Rare words, keyword probability, and text complexity metrics).
""")

email_input = st.text_area("Paste email content here:", height=200)

if st.button("Analyze Email"):
    if not email_input:
        st.warning("Please enter text.")
    else:
        with st.spinner('Processing text and extracting features...'):
            # A. Clean
            cleaned_step_1 = clean_text(email_input)
            
            # B. Preprocess (Lemmatize)
            final_text = preprocess_text_2(cleaned_step_1)
            
            if not final_text:
                st.error("Input text contained only stopwords or was empty after cleaning.")
            else:
                # C. Vectorize Text (TF-IDF)
                tfidf_vector = tfidf.transform([final_text])
                
                # D. Extract Numeric Features
                numeric_df = extract_features(cleaned_step_1, final_text)
                
                # E. Scale Numeric Features
                # Note: Convert to sparse matrix before scaling to match training flow
                numeric_sparse = csr_matrix(numeric_df.values)
                numeric_scaled = scaler.transform(numeric_sparse)
                
                # F. Combine Features (Hstack)
                X_final = hstack([tfidf_vector, numeric_scaled])
                
                # G. Predict
                prediction = model.predict(X_final)[0]
                probability = 0
                
                # Get probability if the model supports it (LogisticReg, LGBM, RF do)
                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(X_final)[0][1]

                # H. Display Output
                st.divider()
                if prediction == 1:
                    st.error("🚨 SPAM DETECTED")
                    if probability:
                        st.write(f"**Confidence:** {probability:.2%}")
                else:
                    st.success("✅ NOT SPAM (HAM)")
                    if probability:
                        st.write(f"**Confidence:** {(1-probability):.2%}")

                # Optional: Show Debug Info
                with st.expander("See Internal Metrics"):
                    st.write("Processed Text:", final_text)
                    st.write("Extracted Numeric Features:", numeric_df)