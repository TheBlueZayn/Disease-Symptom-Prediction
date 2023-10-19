import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import time

from joblib import dump,load
ml_model=load('rf_model.joblib')


# Load Data
df = pd.read_csv("empty_df.csv")
df_2 = pd.read_csv("symptom_Description.csv")
df_3 = pd.read_csv("symptom_precaution.csv")





image = Image.open("ml.jpg")


# Create subsections 
header = st.container()

with header:
    st.title("BlueZayn's Disease Prediction Model")
    st.image(image)
    st.markdown("Takes in symptoms and predicts a disease, gives a description and some precautions to take, asks some questions and generates a summary report that can be downloaded. (*For mobile users, open side panel to input symptoms*)")


st.sidebar.header("What are your symptoms?")
st.sidebar.markdown("Select your symptoms")
st.sidebar.markdown("**Press the predict button when done**")

disease_classes = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
       'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
       'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
       'Common Cold', 'Dengue', 'Diabetes ',
       'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
       'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
       'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 
       'No disease','Osteoarthristis', 'Paralysis (brain hemorrhage)',
       'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
       'Typhoid', 'Urinary tract infection', 'Varicose veins',
       'hepatitis A']
options = st.sidebar.multiselect(" ", options= df.columns)

#Display symptoms dataframe
symptomp = st.container()

with symptomp:
    st.subheader("Your Symptoms")
    for option in options:
        st.write(option)
    for option in options:
        df[option] = 1


#Display predicted disease
# st.subheader("Predicted Disease")

df_list = df.iloc[0].to_list()
df_arr = np.array(df_list)
df_arr = df_arr.reshape(1,-1)

# Make the prediction
prediction = ml_model.predict(df_arr)
dis = prediction[0]
d = disease_classes[dis]

# Display description of predicted disease
descr = (df_2[df_2["Disease"] == d]["Description"].item())

# Display Precautions to take
prec_1 = df_3[df_3["Disease"] == d]["Precaution_1"].item().title()
prec_2 = df_3[df_3["Disease"] == d]["Precaution_2"].item().title()
prec_3 = df_3[df_3["Disease"] == d]["Precaution_3"].item().title()
prec_4 = df_3[df_3["Disease"] == d]["Precaution_4"].item().title()

# Make submit button (This makes the prediction faster and load at once)
if st.sidebar.button('Predict'):
    st.subheader("Predicted Disease")
    st.write(d)

    st.subheader("Description of Predicted Disease")
    st.write(descr)

    st.subheader("Precautions to take")
    st.write(prec_1)
    st.write(prec_2)
    st.write(prec_3)
    st.write(prec_4)
else:
    st.sidebar.markdown("Waiting for your symptoms!")  

# Ask question
# questions = st.container()
# with questions:
summary_early =  (f"""
    Thank you for using our predictor model. 
    Here is a summary
    Your symptoms are: {options} 
    
    Description of predicted disease: {descr}
    
    Precautions to take: 1. {prec_1}
                            2. {prec_2}
                            3. {prec_3} 
                            4. {prec_4}
""")
st.subheader("**Would you like to answer some questions?**")

st.markdown("If no, press the button below")
st.download_button("Download your report", data=summary_early, key=1)

# On real disease if known
st.subheader("Disease History")
st.markdown("We would love to know your real disease to see how well our model did.")
diseas = st.radio("Is your disease known?", options=["not known", "known"])
if diseas != "not known":
    st.text_input("Name of disease", placeholder="type here")

# On medications, if on any
st.subheader("Medication History?")
st.markdown("If you are on any medication, please list out the drugs below. E.g  *Paracetamol, Vitamin C and Loratadine*.")
drugs = st.radio("Are you on any drugs ?",options=["no","yes"])
if drugs == "yes":
    st.text_input("Name of drugs", placeholder="type here")

summary = (f"""
    Thank you for using our predictor model. 
    Here is a summary
    Your symptoms are: {options} 
    
    Your predicted disease is {d}, your real disease is {diseas} and you are on {drugs} drugs
    
    Description of predicted disease: {descr}
    
    Precautions to take: 1. {prec_1}
                            2. {prec_2}
                            3. {prec_3} 
                            4. {prec_4}
""")
st.markdown("download your summary below")
st.download_button("Download your report", data=summary, key=2)
st.markdown("Thank you for trying out the model!")
st.markdown("Want to know how I built this? Check out the codes [here](https://disease-symptom-prediction-kqpnytmyfqmmtbjyxlvkcy.streamlit.app/)")
# else:
    #     st.markdown("Thank you for trying out the model!")
    #     st.markdown("Want to know how I built this? Check out the codes here")
