import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Sale App

This app predicts the **Advertising Sale** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Tv = st.sidebar.slider('Tv', 10.0, 235.0, 17.2)
    Radio = st.sidebar.slider('Radio', 10.0, 46.0, 39.3)
    Newspaper = st.sidebar.slider('Newspaper', 40.0, 70.0, 69.3)
    data = {'Tv': Tv,
            'Radio': Radio,
            'Newspaper': Newspaper}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertising.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
