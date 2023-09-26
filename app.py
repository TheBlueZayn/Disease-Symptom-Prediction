import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Create subsections 
header = st.container()
dataset = st.container()
syptomps = st.container()



with header:
    st.title("BlueZayn's Disease Prediction Model")


with syptomps:
    st.header("What symptoms are you experiencing?")
    st.markdown("Select the symptoms below")
     # Create input colums
    sel_col_1, disp_col_1 = st.columns(2) 

    symp_1 = sel_col_1.selectbox("Symptoms", options=["rash", "Vomiting", "headache", "cough"])

    sel_col_2, disp_col = st.columns(2) 
    symp_2 = sel_col_2.selectbox("More Symptoms", options=["body_rash", "Vomiting", "headache", "cough"])

    
    #sel_col_3, disp_col_3 = st.columns(2) 
    #symp_3 = sel_col_3.selectbox("Symptoms", options=["rash", "Vomiting", "headach", "cough"])




    disp_col_1.write("Your symptoms are:")
    disp_col_1.write(symp_1)
    disp_col_1.write(symp_2)