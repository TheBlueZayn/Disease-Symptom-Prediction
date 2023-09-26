import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Create subsections 
header = st.container()
dataset = st.container()
syptoms = st.container()



with header:
    st.title("BlueZayn's Disease Prediction Model")