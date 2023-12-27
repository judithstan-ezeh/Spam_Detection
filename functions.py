from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords
stemmer = PorterStemmer()

def stemming(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Stem each word in the text
    return ' '.join([stemmer.stem(word) for word in words])

def process_text(text):
    # Remove punctuation
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    
    # Remove stopwords
    return ' '.join([word for word in no_punc.split() if word.lower() not in stopwords.words('english')])


# encoding the text data
def vectorize_text(text):
    vectorizer = CountVectorizer()
    # Preprocess the text first
    preprocessed_text = process_text(text)
    # Use `transform` not `fit_transform`
    return vectorizer.transform([preprocessed_text])
