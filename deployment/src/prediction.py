# main library for streamlit
import streamlit as st

# import libraries for prediction inference
import pickle
import pandas as pd
import numpy as np


# load the model file (from src/)
## alongwith the pipeline
with open('./src/term_depo_predictor.pkl', 'rb') as file:
    best_svm = pickle.load(file)


# define same function on the main notebook

## function for re-grouping column `job`
def job_top5_grouping(df, col='job'):
    
    # list the top 5 from such column
    job_top5 = ['blue-collar', 'management', 'technician', 'admin.', 'services']

    # change other values into 'other'
    df.loc[~(df[col].isin(job_top5)), col] = 'other'   

## function for re-grouping column `day`
def day_binning(series):
    return pd.cut(
        series,
        bins=[0, 6, 12, 18, 24, 31],
        labels=['1 - 6', '7 - 12', '13 - 18', '19 - 24', '25 - 31']
    )

## function for re-grouping column `month`
def month_binning(series):

    # map the month to number first
    month_number = {
        'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4,
        'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8,
        'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12
    }

    # map and replace column `month` with number
    series = series.map(month_number)

    return pd.cut(
        series,
        bins=[0, 3, 6, 9, 12],
        labels=['jan - mar', 'apr - jun', 'jul - sep', 'oct - dec']
    )


# for run all in the function below
def run():
    
    # create title
    st.title('Data-Driven Term Deposit Subscription Modeling for Marketing Campaigns')
    st.write('## Prediction Based on Input Data')
    st.write('*Created by: Fernando Namora*')
    st.write('''
             This project is to build a model to predict 
             the client will subscribe a term deposit 
             based on bank marketing campaigns historical data. 
            ''')
    
    # add source data
    st.markdown(
        """
        <small>
        Source of Dataset :
        <a href="https://archive.ics.uci.edu/dataset/222/bank+marketing">
        UC-Irvine
        </a>
        </small>
        """,
        unsafe_allow_html=True
    )

    # separate line
    st.markdown('---')


    # create form
    with st.form(key='term-deposit-subscription-model'):
        
        st.write('### Client Details Data')
        ## optional input
        name = st.text_input('Client name (optional)', value = '**--input-name--**',
                             help = 'This name is only use as an index, if not inputted will use "Client" as index in the DataFrame')

        ## client details data
        age = st.slider('Age', value = 18, 
                              min_value = 18, max_value = 95,
                              help = 'Age of the client',
                              step = 1)
        
        # create two columns
        col1, col2 = st.columns(2)

        with col1: # left columns

            job = st.selectbox('Type of job of the client', 
                            (
                                'blue-collar', 'management', 'technician',
                                'admin.', 'services', 'retired',
                                'self-employed', 'entrepreneur',
                                'unemployed', 'housemaid',
                                'student', 'unknown'
                                ), 
                            index = 0)
            
            marital = st.selectbox(
                'Marital status', ('married', 'single', 'divorced'), index = 0
            )

        with col2: # right columns
            education = st.selectbox(
                'Education level', ('primary','secondary','tertiary','unknown'), index = 0
            )
            
            balance = st.number_input('Balance', value = 100, 
                                min_value = -9999, max_value = 199999,
                                help = 'Average yearly balance of the client',
                                step = 1)

        # create three columns
        col1, col2, col3 = st.columns(3)

        with col1: # left column
            default = st.radio(
                'The client has credit in default ?', ('yes', 'no'), index = 0
            )
        
        with col2: # mid column
            housing = st.radio(
                'The client has housing loan ?', ('yes', 'no'), index = 0
            )
        
        with col3: # right column
            loan = st.radio(
                'The client has personal loan ?', ('yes', 'no'), index = 0
            )

        ## last contact of the current campaign details
        st.write('### Current Campaign Details')

        # create two columns
        col1, col2 = st.columns(2)

        with col1: # left column
            contact = st.selectbox(
                'Contact communication type', ('cellular', 'telephone', 'unknown'), index = 0
            )

        with col2: # right column
            month = st.select_slider(
                'Last contact month', 
                (
                    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
                ), 
                help = 'Last contact month of the year on current campaign',
                value = 'jan'
            )

        day = st.slider(
            'Last contact day', 
            help = 'Last contact day of the month on current campaign',
            value = 1,min_value = 1, max_value = 31
        )

        # create two columns
        col1, col2 = st.columns(2)

        with col1: # left column
            duration = st.selectbox(
            'The duration in second of **this** call data : ', 
            ('**--missing--**'),
            help = 'Because the call to this client is not performed yet',
            index = 0
            )

        with col2: # right column
            campaign = st.number_input('Number of calls to this client : ', value = 0, 
                                min_value = 0, max_value = 999,
                                help = '''Number of contacts performed during this campaign 
                                and for this client''',
                                step = 1)

        ## past campaign details
        st.write('### Past Campaigns Details')

        # create three columns
        col1, col2, col3 = st.columns(3)

        with col1: # left column
            pdays = st.number_input('Days after client last contacted from **previous** campaigns :', value = -1, 
                                min_value = -1, max_value = 999,
                                help = '''Number of days that passed by after the client was 
                                last contacted from **previous** campaign. If this client was 
                                not previously contacted, the default value is -1''',
                                step = 1)
        
        with col2: # mid column
            previous = st.number_input('Number of calls to this client on **previous** campaigns :', value = 0, 
                                min_value = 0, max_value = 999,
                                help = '''Number of contacts performed for this client
                                and on **previous** campaign''',
                                step = 1)
        
        with col3: # right column
            poutcome = st.selectbox(
                'Outcome of the **previous** marketing campaign', 
                ('success', 'failure', 'other', 'unknown'), index = 0
            )
       
        submitted = st.form_submit_button('Predict')
    
    st.markdown('---') # line separation
    
    # handling value
    if name == '**--input-name--**':
        name = 'Client'
    if duration == '**--missing--**':
        duration = np.nan

    # create dataframe of data inference
    data_inf = pd.DataFrame({
        'age' : age,
        'job' : job,
        'marital' : marital,
        'education' : education,
        'default' : default,
        'balance' : balance,
        'housing' : housing,
        'loan' : loan,
        'contact' : contact,
        'day' : day,
        'month' : month,
        'duration' : duration,
        'campaign' : campaign,
        'pdays' : pdays,
        'previous' : previous,
        'poutcome' : poutcome
    }, index = [name])

    if submitted:
        st.write('### The Input Data:')
        st.dataframe(data_inf)
        
        st.markdown('---') # line separation

        # regroup column `job`, `day`, and `month`
        job_top5_grouping(data_inf, 'job')
        data_inf['day'] = day_binning(data_inf['day'])
        data_inf['month'] = month_binning(data_inf['month'])

        # load to the pipeline then predict
        y_pred_inf = best_svm.predict(data_inf)
        
        # create dataframe
        y_pred_inf_df = pd.DataFrame(y_pred_inf, index = [name])
        y_pred_inf_df.columns = [
            'Will the client subscribe a term deposit?'
        ]

        # show the results
        st.write('### Prediction Output:')
        st.dataframe(y_pred_inf_df)

        # condition
        if y_pred_inf_df[
            'Will the client subscribe a term deposit?'
        ][name] == 0:
            st.write('''It is unfortunate that **this client is predicted 
                     will not subscribe** a term deposit products from the Bank.''')
        else:
            st.write('''The good news is **this client is predicted will subscribe** 
                     a term deposit product, so it is worth for the next or current campaign 
                     to further approach this client.''')


if __name__ == '__main__':
    run()