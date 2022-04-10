import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime
import datetime as dt
from  keras.models import load_model
import streamlit as st
import string
import yahoofinance as yf
from streamlit_option_menu import option_menu




def p():
    st.title('Stock Price Prediction')

    stocks = pd.read_csv('ticker.csv')
    selected_stocks = st.selectbox("Selected stocks for prediction", stocks)

    # user_input=st.text_input('Enter Stock Ticker','^NSEI')
    start = st.date_input('start date', dt.date(2005, 1, 1))
    # start =st.text_input('Enter starting date','2005-01-01')
    end = st.date_input('END date', dt.date.today())
    df = web.DataReader(selected_stocks, 'yahoo', start, end)
    st.subheader("Data from ")
    st.write(df)

    # visualtion

    st.subheader('closing Price vs Time Graph')

    fig = plt.figure(figsize=(18, 12))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('closing Price vs Time Graph with 100DMA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('closing Price vs Time Graph with 100DMA &200DMA')

    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma200, 'r')
    plt.plot(ma100, 'y')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    # splitting data into training ionto testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i - 100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # ml model

    model = load_model('G_keras_model.hs')

    past_100days = data_training.tail(100)
    final_df = past_100days.append(data_testing, ignore_index=True)
    final_df.head()

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # making prediction
    y_predicted = model.predict(x_test)

    scale_factor = 1 / scaler.scale_[0]
    y_test = y_test * scale_factor
    y_predicted = y_predicted * scale_factor

    # final graph

    # final_data={y_test,y_predicted}

    final_data_frame = pd.DataFrame(list(zip(y_test, y_predicted)))
    st.write("Test and predicted Data")
    st.write(final_data_frame)
    st.subheader("predicted vs original")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='original Price')
    plt.plot(y_predicted, 'r', label='Pridicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)



selected = option_menu(menu_title="Main Menu",options=["Home","Financial","Contact"],

                         orientation="horizontal"
                         )

if selected =="Home":
    p()
elif selected =="Financial":
  st.title(f"Welcome to {selected}")
  st.write(m())
elif selected =="Contact":
  st.title(f"Welcome to {selected}")

else :
    st.title(" ")

