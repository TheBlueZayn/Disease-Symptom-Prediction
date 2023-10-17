import streamlit as st
import pandas as pd
import numpy as np


st.header("Test run")

options = st.selectbox("syptomps", options=["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing"])

#options = options
symp = []
symp.append(options)
st.write(symp)


df = pd.read_csv("empty_df.csv")
df[options] = 1
st.write(df)