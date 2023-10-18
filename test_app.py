import streamlit as st
import pandas as pd
import numpy as np
from joblib import dump,load
ml_model=load('rf_model.joblib')
df = pd.read_csv("empty_df.csv")

disease_classes = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
       'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
       'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
       'Common Cold', 'Dengue', 'Diabetes ',
       'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
       'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
       'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
       'Osteoarthristis', 'Paralysis (brain hemorrhage)',
       'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
       'Typhoid', 'Urinary tract infection', 'Varicose veins',
       'hepatitis A']

st.header("Test run")
options = st.multiselect("Syptoms", options= df.columns)
    
st.write("Your symptoms are:")
for option in options:
    st.write(option)
for option in options:
    df[option] = 1

st.write(df)

df_list = df.iloc[0].to_list()

df_arr = np.array(df_list)
df_arr = df_arr.reshape(1,-1)

# Make the prediction
prediction = ml_model.predict(df_arr)
prediction
dis = prediction[0]
d = disease_classes[dis]

st.write(d)


