
# kaggle dataset link "https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbi1NLXFrSS1IZEpKT1Z2QnI3cmUxeWRwNW5rZ3xBQ3Jtc0ttbU1FTjhTTGFBVWhqcG1vbmM0TXJoZ290azhOajM3bl93bmxDdGhRMXRHd0NyMzB6MGNGeWtjWDhyOVV4RUk2N0k5TWJVTGJnSXFzdmJOd3NYMTlKNy00alJBME81UVhaM3A1T0N4akljSVBTLWl6NA&q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fmlg-ulb%2Fcreditcardfraud&v=239TaYSQI-s"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import streamlit as st

df=pd.read_csv('creditcard.csv')
legit=df[df.Class==0]
fraud=df[df.Class==1]
print(fraud)
legit_sample=legit.sample(n=len(fraud),random_state=2)
df=pd.concat([legit_sample,fraud],axis=0)
x=df.drop(columns="Class",axis=1)
y=df["Class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
train_accuracy=accuracy_score(model.predict(x_train), y_train)
test_accuracy=accuracy_score(model.predict(x_test), y_test)
tot=accuracy_score(y_test,y_pred)
print(train_accuracy)
print(test_accuracy)
print(tot)


st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

input_df=st.text_input("input all features")
split_input=input_df.split(',')
submit=st.button("submit")


if submit:
    features=np.array(split_input,dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))
    if prediction[0]==0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")