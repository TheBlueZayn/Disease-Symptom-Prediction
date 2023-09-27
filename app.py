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
    symp_1 = st.sidebar.selectbox("Symptom_1", options= ["rash" , "headache", "vomiting"])


    data = {"Symptom_1": symp_1}

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
d = d.strip().replace(" ", "_")

st.write("The predicted Disease is")
st.write(d)

# Display description of predicted disease
st.subheader("Desription of Predicted Disease")

# Display Precautions to take
st.subheader("Precautions to take!")