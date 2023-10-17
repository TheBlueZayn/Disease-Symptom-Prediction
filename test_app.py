import streamlit as st
import pandas as pd
import numpy as np


st.header("Test run")

options = st.multiselect("syptomps", options=["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing"])

#options = options

st.write(options)


df = pd.read_csv("empty_df.csv")
st.write(df)