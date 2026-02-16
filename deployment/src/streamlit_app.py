import streamlit as st
import eda 
import prediction 

st.set_page_config(
    page_title = 'Machine learning model to predict customer subscription for term deposit campaigns',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

page = st.sidebar.selectbox('Choose page : ',
                            ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()

else:
    prediction.run()