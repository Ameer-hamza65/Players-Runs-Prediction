import numpy as np
import pandas as pd
import streamlit as st

data=pd.read_csv("test.csv")

data.drop(columns=['Unnamed: 11','Player'],axis=1,inplace=True)
data.drop('Unnamed: 0',axis=1,inplace=True)
data.drop('Span',axis=1,inplace=True)

data = data.rename(columns={'100': 'Hundred', '50': 'Fifty', '0': 'Duck'})


# Remove rows where 'Inns' is '-'
data = data[data['Inns'] != '-']



data['Inns'] = data['Inns'].astype('int64')
data['NO']=data['NO'].astype('int64')
data['Runs']=data['Runs'].astype('int64')
data['HS'] = data['HS'].str.replace('*', '', regex=False)
data['HS']=data['HS'].astype('int64')
# Remove rows where 'Inns' is '-'
data = data[data['Ave'] != '-']

data['Ave']=data['Ave'].astype('float64')
data['Hundred']=data['Hundred'].astype('int64')
data['Fifty']=data['Fifty'].astype('int64')
data['Duck']=data['Duck'].astype('int64')


x=data.drop('Runs',axis=1)
y=data.iloc[:,3]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x_train,y_train)


# Using HTML to style the title
st.markdown("<h1 style='color:#b80254;'>Cricketer Runs Predictor</h1>", unsafe_allow_html=True)

st.write("")

col1, col2, col3 = st.columns(3)

# Define the widgets in each column
with col1:
    Mat = st.number_input("Matches", min_value=0, step=1, format="%d")
    Inns = st.number_input("Innings", min_value=0, step=1, format="%d")
    NO = st.number_input("Not out", min_value=0, step=1, format="%d")

with col2:
    Hundred = st.number_input("Hundreds", min_value=0, step=1, format="%d")
    Fifty = st.number_input("Fifties", min_value=0, step=1, format="%d")
    Duck = st.number_input("Number of Ducks", min_value=0, step=1, format="%d")

with col3:
    HS = st.number_input("Highest Score", min_value=0, step=1, format="%d")
    Ave = st.number_input("Average", min_value=0.0, format="%f")


def predict_runs(Mat, Inns, NO, HS, Ave, Hundred, Fifty, Duck):
    input_data=[[Mat, Inns, NO, HS, Ave, Hundred, Fifty, Duck]]
    prediction=rf.predict(input_data)
    return prediction[0]


if st.button("Predict"):
    prediction=predict_runs(Mat, Inns, NO, HS, Ave, Hundred, Fifty, Duck)
    st.write("")
    prediction_int = int(prediction)

# Displaying the formatted text
    st.markdown(f"**Batsman runs are: <span style='font-size:24px'>{prediction_int}</span>**", unsafe_allow_html=True)
    
