import matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests as requests
from datetime import date
import tensorflow as tf
matplotlib.use('Agg')
import plotly.graph_objs as go
from lstmWithOneVariable import run_lstm_model
from lstmWithThreeVariable import run_lstm_model_multiVariable

api_key = 'FU7G5GQKP96N4PJV'

st.set_page_config(page_title="MyFirstStreamlitPage", layout="wide")
st.title("STOCK PREDICTION WITH LSTM")

#Settings
test_df = pd.read_csv('stock_tickerListing.csv')

justsymbol = test_df["symbol"]
START = date(2020,1,1).strftime("%Y-%m-%d")
TODAY = date.today().strftime("%Y-%m-%d")
period = 365

company_name = "-"
description = "-"
company_country = "-"
data = None

st.info("After selecting a stock ticker. User can select a variable from dropdown list which will apply to Single Variable LSTM model.\n\n For Multiple Variables LSTM model due to hardware and time taken limitations the variables have been fixed.")
#selection boxes
col1, col2, col_empty = st.columns([2,2,3])
col3, col4 = st.columns([1,1])

selected_ticker = col1.selectbox('Select a ticker', justsymbol)

predictOn = col2.selectbox("Predict based on", ("Open", "Close", "High", "Low"))

# Alpha Vantage API call
info_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={selected_ticker}&apikey={api_key}'
info_r = requests.get(info_url)
info_data = info_r.json()
company_name = info_data['Name']
description = info_data['Description']
company_country = info_data['Country']

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
data = load_data(selected_ticker)
data.index = data.pop("Date")

st.subheader('Company Overview')
col_name, col_country = st.columns([2,2])
col_name.write(f'Company : {company_name}')
col_country.write(f'Country : {company_country}')
st.write(f'Background :\n\n{description}')

st.subheader('Historical Records')

if company_name and company_name != '-':
    st.write(f'Raw data of {company_name}')
else:
    st.warning('Please click confirm to search for info')

def plot_raw_data():
    if data is not None:
        
        col5, col6 = st.columns([5,5])
        col5.dataframe(data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close", line=dict(color='blue')))
        fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
        col6.plotly_chart(fig, use_container_width=True, config={'responsive': True})

plot_raw_data()

#After user click this button only do searching
if col3.button("Click to run LSTM prediction model"):

    def plot_prediction_data():
        if data is not None:
            st.subheader(f'{company_name} Stock Price Prediction with Single Variable LSTM Model')
            with st.spinner("Running calculation be patient.."):
                aFig = run_lstm_model(data, predictOn)
                st.pyplot(aFig)
            st.success("Single Variables Prediction Complete!")

            st.subheader(f'{company_name} Stock Price Prediction with Multiple Variables LSTM Model')
            with st.spinner("Running calculation be patient.."):
                bFig = run_lstm_model_multiVariable(data)
                st.pyplot(bFig)
            st.success("Multiple Variables Prediction Complete!")
    
    plot_prediction_data()


