# importing necessary libraries
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import string
from sklearn.feature_extraction.text import CountVectorizer

import pickle

# importing the function file for preprocessing the text
from functions import *

import streamlit as st


# Custom Theme
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.streamlit.io',
        'Report a bug': "https://github.com",
        'About': "# This is a Spam Detection App!"
    }
)

# Loading the trained model
# @st.cache
def load_model():
    with open('logistic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Preprocess the text
def process_text(text):
    stemmer = PorterStemmer()

    # Remove punctuation
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    # Remove stopwords and apply stemming
    return ' '.join([stemmer.stem(word) for word in no_punc.split() if word.lower() not in stopwords.words('english')])

# Load the model
model = load_model()

# Load your CountVectorizer
@st.cache_data
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

vectorizer = load_vectorizer()



# Streamlit app with enhancements
def main():
        # Use Streamlit's theme options to customize colors, fonts, etc.
    st.markdown("""
        <style>
        .main {
            background-color: #F5F5F5;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 10px;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            transition-duration: 0.4s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True)
    st.title("Spam Detection App")
    
    st.markdown("""
        Welcome to the Spam Detection App! Enter any text below and click 'Predict' to see if it's spam.
        Great for filtering out unwanted messages and emails.
    """)

    # Layout enhancements
    with st.spinner('Processing...'):
        text = st.text_area("Enter Text to Analyze for Spam:", "Type your message here...")

        if st.button("Predict"):
            processed_text = process_text(text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)

            # Visual feedback
            if prediction == 0:
                st.success("This text is Not Spam.")
            else:
                st.error("This text is Spam.")

    # Sidebar for additional information or settings
    with st.sidebar:
        st.info("About the App")
        st.write("""
            This app uses a machine learning model to detect spam in text messages or emails.
            Simply enter the text and let the model predict its nature.
        """)
        # Add developer's name at the bottom of the sidebar
        st.markdown("---")  # This adds a horizontal line for separation
        st.markdown("Developed by [Judith Ofoedu]")

if __name__ == "__main__":
    main()
