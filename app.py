import pickle
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources if necessary
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))

# Load the model and the TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Define the function to clean the text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Define the prediction function
def predict_sentiment(review):
    cleaned = clean_text(review)
    vect = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vect)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Streamlit UI
st.title("Sentiment Analyzing System")

# Create a text area for user input
user_input = st.text_area("Enter your review:", height=150)

# Create a button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        if sentiment == 'Positive':
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review to analyze.")

# Add custom CSS for enhanced appearance
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #4CAF50, #45a049);
}
.stTextArea > div > div > textarea {
    font-size: 16px;
    border-radius: 10px;
    border: 2px solid #4CAF50;
    transition: border-color 0.3s;
}
.stTextArea > div > div > textarea:focus {
    border-color: #FF5733; /* Change border color on focus */
}
.stButton > button {
    font-size: 18px;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    transition: background-color 0.3s, transform 0.3s;
}
.stButton > button:hover {
    background-color: #45a049; /* Darker green on hover */
    transform: scale(1.05); /* Slightly enlarge button on hover */
}
.stTitle {
    color: white;
    text-shadow: 2px 2px 4px #000000;
}
</style>
""", unsafe_allow_html=True)