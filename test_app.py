import streamlit as st
import pandas as pd
import numpy as np


st.header("Test run")

options = st.selectbox("syptomps", options=["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing"])
symp_1 = st.selectbox("Symptomps", options= ["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes"])
#options = options
# symp = []
# symp.append(options)

st.write(options)


df = pd.read_csv("empty_df.csv")
df[options] = 1
df[symp_1]= 1
st.write(df)