import streamlit as st
import requests as requests

st.set_page_config(page_title="MyFirstStreamlitPage", layout="wide")

st.title("Streamlit Web App")
st.sidebar.success("Select a page above.")
st.sidebar.header("Details")

st.sidebar.write("Name : YEONG WEI HAO")
st.sidebar.write("Student No : 041180354")
st.sidebar.write("Course Code : TIS311/04")
st.sidebar.write("Subject : IS Project")
st.sidebar.write("Class Code : IS14-SEP22")
st.sidebar.write("Assignment No : Assignment 3")
st.sidebar.write("Tutor's name Dr. Manjeevan Singh Seera")
st.sidebar.write("Course Lead's name : Dr. Vimala")

st.subheader("Introduction")
st.info("Description: This is a web application created with Streamlit and integrated with LSTM model. It able to do stock prediction based on user selected stock.")
st.warning("WARNING: This web application is solely for academic purposes. Prediction result only for references.")

st.subheader("Libraries :")

style = """
    <style>
        .box {
            background-color: #F0F0F0;
            padding: 0px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .column-content {
            font-size: 24px;
        }
    </style>
"""
st.markdown(style, unsafe_allow_html=True)

# Create columns inside the box
with st.container():
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    col5, col6, col7, col8 = st.columns([1, 1, 1, 1])

    # Add content to columns with increased font size
    col1.markdown("<p class='column-content'><a href='https://www.alphavantage.co/'>Alpha Vantage</a></p>", unsafe_allow_html=True)
    col2.markdown("<p class='column-content'><a href='https://pypi.org/project/matplotlib/'>Matplotlib v3.7.1</a></p>", unsafe_allow_html=True)
    col3.markdown("<p class='column-content'><a href='https://pypi.org/project/numpy/'>Numpy v1.24.3</a></p>", unsafe_allow_html=True)
    col4.markdown("<p class='column-content'><a href='https://pypi.org/project/pandas/'>Pandas v1.5.2</a></p>", unsafe_allow_html=True)
    col5.markdown("<p class='column-content'><a href='https://pypi.org/project/scikit-learn/'>Scikit-learn v1.3.0</a></p>", unsafe_allow_html=True)
    col6.markdown("<p class='column-content'><a href='https://docs.streamlit.io/library/get-started'>Streamlit v1.24.1</a></p>", unsafe_allow_html=True)
    col7.markdown("<p class='column-content'><a href='https://pypi.org/project/tensorflow/'>Tensorflow v2.13.0</a></p>", unsafe_allow_html=True)
    col8.markdown("<p class='column-content'><a href='https://pypi.org/project/yfinance/'>Yahoo Finance</a></p>", unsafe_allow_html=True)


