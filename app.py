import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

model = load_model(r'C:\Users\vidhi\OneDrive\Desktop\STOCK\Stock Predictions Model.keras')


st.header("Stocks Price Predictor")

stock = st.text_input('Enter Stock Symbol', "GOOG")
start= '2018-02-05'
end= '2022-02-05'
data = yf.download(stock,start,end)

st.subheader("Stock Data")
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

scalar = MinMaxScaler(feature_range=(0,1))

past_50_days = data_train.tail(50)
past_50_days = pd.DataFrame(past_50_days, columns=['Close'])
data_test = pd.concat([past_50_days,data_test], ignore_index = True)
data_test_scale = scalar.fit_transform(data_test)

st.subheader("Price vs moving average of 50 days")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label='Close Price', color='green')
plt.plot(ma_50_days, label='50-Day Moving Average', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader("Price vs moving average of 50 days and 100 days")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label='Close Price', color='green')
plt.plot(ma_50_days, label='50-Day Moving Average', color='red')
plt.plot(ma_100_days, label='100-Day Moving Average', color='blue')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)

x = []
y = []

for i in range(50, data_test_scale.shape[0]):
    x.append(data_test_scale[i-50:i])  
    y.append(data_test_scale[i, 0])  

x = np.array(x)
y = np.array(y)

predict = model.predict(x)

scale = 1/scalar.scale_

predict = predict *scale

y= y*scale

st.subheader('Stock Price, Predicted vs Actual')
fig3=plt.figure(figsize=(8,6))
plt.plot(y, color='blue', label='Actual Close Price')
plt.plot(predict, color='red', label='Predicted Close Price')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()
st.pyplot(fig3)