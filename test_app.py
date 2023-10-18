import streamlit as st
import pandas as pd
import numpy as np
import ast 

st.header("Test run")
df = pd.read_csv("empty_df.csv")
options = st.multiselect("Syptoms", options= df.columns)
    
st.write("Your symptoms are:")
for option in options:
    st.write(option)
    df[option] = 1

st.write(df)