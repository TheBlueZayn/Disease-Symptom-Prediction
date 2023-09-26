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
    st.markdown("Select the symptomps below")
     # Create input colums
    sel_col, disp_col = st.columns(2)   
    symp = sel_col.selectbox("Symptoms", options=["rash", "Vomiting", "headach"])