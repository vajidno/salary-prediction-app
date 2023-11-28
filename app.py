import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np


data = pd.read_csv("Salary_Data.csv")
x = np.array(data["YearsExperience"]).reshape(-1,1)
y = np.array(data["Salary"]).reshape(-1,1)
model = LinearRegression()
model.fit(x,y)


st.title("Salary Predictor")
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])

if nav == "Home":
    st.image("sal.jpg",width = 700)
    if st.checkbox("show table"):
        st.table(data)

    if st.checkbox("Graph"):
        plt.figure(figsize=(10,5))
        plt.scatter(data["YearsExperience"],data["Salary"])
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot()

if nav == "Prediction":
    st.header("Know Your Salary")
    val = st.number_input("Enter Your Experience",0.00,20.00,step = 0.25)
    val = np.array(val).reshape(-1,1)
    pred = model.predict(val)
    if st.button("predict"):
        st.success(f"Your predicted Salary is {pred}")

if nav == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Experience",0.0,20.0)
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)
    if st.button("submit"):
        to_add = {"YearsExperience":[ex],"Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("Salary_Data.csv",mode='a',header = False,index= False)
        st.success("Submitted")

