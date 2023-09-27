import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Load Data
df_1 = pd.read_csv("dataset.csv")
df_2 = pd.read_csv("symptom_Description.csv")
df_3 = pd.read_csv("symptom_precaution.csv")
df_4 = pd.read_csv("Symptom-severity.csv")

# Encode symptomps with severity
def encode_symptoms(df, df_4):
    for i in df_4.index:
        symptom = df_4["Symptom"][i]
        weight = df_4["weight"][i]
        df = df.replace(symptom, weight)

    # Replace missing values with 0
    df = df.fillna(0)

    # Additional hardcoded replacements
    df = df.replace("foul_smell_of_urine", 5)
    df = df.replace("dischromic__patches", 6)
    df = df.replace("spotting__urination", 6)
    
    return df

# Encode data with severity
new_df_1 = encode_symptoms(df_1, df_4)

# separating the data and labels
X = new_df_1.drop(columns='Disease', axis=1)
Y = new_df_1['Disease']

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
# Set standardized data
standardized_data = scaler.transform(X)
X = standardized_data
Y = new_df_1['Disease']

# Train Model
from sklearn.ensemble import RandomForestClassifier
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rfc_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
rfc_classifier.fit(X_train, Y_train)


# Create subsections 
header = st.container()
# dataset = st.container()
# syptomps = st.container()



with header:
    st.title("BlueZayn's Disease Prediction Model")
    st.markdown("Takes in symptoms and predicts a  disease, it's description and precautions")


st.sidebar.header("What are your symptoms?")

def user_input_symp():
    symp_1 = st.sidebar.selectbox("Symptom_1", options= [np.nan, "skin_rash" , "headache", "vomiting"])
    symp_2 = st.sidebar.selectbox("Symptom_2", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_3 = st.sidebar.selectbox("Symptom_3", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_4 = st.sidebar.selectbox("Symptom_4", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_5 = st.sidebar.selectbox("Symptom_5", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_6 = st.sidebar.selectbox("Symptom_6", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_7 = st.sidebar.selectbox("Symptom_7", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_8 = st.sidebar.selectbox("Symptom_8", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_9 = st.sidebar.selectbox("Symptom_9", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_10 = st.sidebar.selectbox("Symptom_10", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_11 = st.sidebar.selectbox("Symptom_11", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_12 = st.sidebar.selectbox("Symptom_12", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_13 = st.sidebar.selectbox("Symptom_13", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_14 = st.sidebar.selectbox("Symptom_14", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_15 = st.sidebar.selectbox("Symptom_15", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_16 = st.sidebar.selectbox("Symptom_16", options= [np.nan,"skin_rash" , "headache", "vomiting"])
    symp_17 = st.sidebar.selectbox("Symptom_17", options= [np.nan,"skin_rash" , "headache", "vomiting"])


    data = {"Symptom_1": symp_1, "Symptom_2": symp_2, "Symptom_3": symp_3, "Symptom_4": symp_4, "Symptom_5": symp_5, "Symptom_6": symp_6, "Symptom_7": symp_7, "Symptom_8": symp_8, "Symptom_9": symp_9, "Symptom_10": symp_10, "Symptom_11": symp_11, "Symptom_12": symp_12, "Symptom_13": symp_13, "Symptom_14": symp_14, "Symptom_15": symp_15, "Symptom_16": symp_16, "Symptom_17": symp_17}

    sympts = pd.DataFrame(data, index=[0])
    return sympts

#Display symptoms dataframe
df = user_input_symp()
st.subheader("Your Symptoms")
st.write(df)

#Display predicted disease
st.subheader("Predicted Disease")

# Predict disease
df = encode_symptoms(df, df_4)
standardized_df = scaler.transform(df)
d = rfc_classifier.predict(standardized_df).item()
d = d.strip().replace("_", " ")

#st.write("The predicted Disease is:")
st.write(d)

# Display description of predicted disease
st.subheader("Desription of Predicted Disease")
descr = (df_2[df_2["Disease"] == d]["Description"].item())
st.write(descr)


# Display Precautions to take
st.subheader("Precautions to take!")
prec_1 = df_3[df_3["Disease"] == d]["Precaution_1"].item().title()
prec_2 = df_3[df_3["Disease"] == d]["Precaution_2"].item().title()
prec_3 = df_3[df_3["Disease"] == d]["Precaution_3"].item().title()
prec_4 = df_3[df_3["Disease"] == d]["Precaution_4"].item().title()

st.write(prec_1)
st.write(prec_2)
st.write(prec_3)
st.write(prec_4)