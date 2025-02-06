import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import streamlit as st
import pickle

# Initialize nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Define the pre-processing function
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_word = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    tokenized_text = text.split()
    tokens = [word for word in tokenized_text if word not in stop_word]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ''.join(tokens)

# Load the dataset
def load_data():
    df = pd.read_csv('/media/anjali/data/AI-ML/learning/NLP/dataset/Emails.csv')

    df.drop('Unnamed: 0', axis=1, inplace=True)
    df['clean_text'] = df['content'].apply(preprocess_text)
    return df

# Define function to train model
def train_model(df):
    encoder = OneHotEncoder()
    encoded_labels = encoder.fit_transform(df[['Class']]).toarray()
    df_encoded = pd.DataFrame(encoded_labels, columns=encoder.get_feature_names_out(['Class']))
    df = pd.concat([df, df_encoded], axis=1)
    df.drop('Class', axis=1, inplace=True)

    vectorizer = TfidfVectorizer(max_features=5000)
    x_tfidf = vectorizer.fit_transform(df['clean_text'])
    y = np.argmax(df_encoded.values, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_tfidf, y, test_size=0.2, random_state=42)
    
    # Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    # Save the trained model
    with open('Email.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model, vectorizer

# Load model and vectorizer if they exist
def load_model():
    try:
        with open('Email.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.warning("Model not found. Training a new model...")
        df = load_data()
        return train_model(df)

# Streamlit UI setup
st.title("Email Spam Classifier")
st.subheader("Upload your Email Data and Get Spam Prediction")

# Show CSV loading option
if st.button("Load CSV File"):
    df = load_data()
    st.write(df.head())
    
# Input form for new email prediction
email_input = st.text_area("Enter the email content")

# Predict button
if st.button("Predict Spam or Not"):
    if email_input:
        model, vectorizer = load_model()
        email_cleaned = preprocess_text(email_input)
        email_vectorized = vectorizer.transform([email_cleaned])
        prediction = model.predict(email_vectorized)
        
        if prediction == 1:
            st.write("This email is **Spam**!")
        else:
            st.write("This email is **Not Spam**!")
    else:
        st.warning("Please enter an email to predict.")
