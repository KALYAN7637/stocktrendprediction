import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader  as data
from keras.models import load_model
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import plotly_express as px
import sklearn as sk

import stocknews
import streamlit as st
import  datetime as dt


from plotly.subplots import make_subplots
import plotly.graph_objects as go




st.title('Stock Trend Prediction')
ticker=st.text_input('Enter Ticker Name','AAPL')

start=st.date_input('Start', value=pd.to_datetime('2010-01-01'))
end=st.date_input('End', value=pd.to_datetime('today'))


df=yf.download(ticker,start,end)
fig= px.line(df,x=df.index, y=df['Close'], title=ticker)
st.plotly_chart(fig)

st.subheader('Data From Start Date To Today')
st.write(df.tail())

st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Open)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#splitting data into training and testing

data_training=pd.DataFrame(df['Close'][0:int(len(df)*.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*.70):int(len(df))])

#print(data_training.shape)
#print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training) 

x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

model=load_model(r'keras_model.h5')


past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing, ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler.scale_

scale_factor=1/scaler.scale_[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor












#final graph


st.subheader('Prediction vs Original Price')
plt2=plt.figure(figsize=(12,6))
plt.plot(y_predicted,'r',label='Predicted Price')
plt.plot(y_test,'b',label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt2)
plt.show()





#pricing_df, fundamental_df, news=st.tabs(['Pricing Data','Fundamental Data','Top 10 News'])

st.subheader('Price Movements')
df2=df
df2['% Change']=df['Close']/df['Close'].shift(1) - 1
df2.dropna(inplace = True)
st.write(df)
annual_return=df2['% Change'].mean()*252*100
st.write('Annual Return is', annual_return,'%')
stdev=np.std(df['% Change'])*np.sqrt(252)
st.write('Standard Deviation is',stdev*100,'%')
st.write('Risk Adj. Return is',annual_return/(stdev*100))
