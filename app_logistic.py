import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from nltk.stem.porter import PorterStemmer

import psycopg2

ps = PorterStemmer()


def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)
def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

vectorizer = pickle.load(open('C:/Users/ckaus/Desktop/Machine Learning/Project/lib/vectorizer_1.pkl','rb'))
model = pickle.load(open('C:/Users/ckaus/Desktop/Machine Learning/Project/lib/model_1.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = text_preprocess(input_sms)
    transform_sms=stemmer(transformed_sms)
    # 2. vectorize
    vector_input = vectorizer.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
