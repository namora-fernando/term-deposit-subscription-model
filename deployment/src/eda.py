# main library for streamlit
import streamlit as st

# import libraries for EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import phik


## phi-k correlation print - use in EDA question 4
### define function to correlate variables with default
def compute_phik_correlation(dataframe, columns, target):
    subset = dataframe[columns]
    correlation_matrix = subset.phik_matrix()
    return correlation_matrix[target]


# for EDA question 1
def EDA_q1(df):
    st.write('### 1. What is the percentage of subscribed vs. non-subscribed customers ?')
    st.markdown(
        '''
        If we look at the frequencies of subscribed and non-subscribed customers in this 
        historical dataset, also see the percentage:
        '''
    )

    ## first part - the dataframe of frequencies
    frequencies = df['y'].value_counts()
    table_print = pd.DataFrame(frequencies)

    ## calculate the percentage - times 100 first
    table_print['percentage'] = (
        table_print['count'] * 100)/(table_print['count'].sum()
    )

    ## then round 2 decimal and change to string and show % at the end
    table_print['percentage'] = table_print['percentage'].round(2).astype(str) + ' %'

    ## show the results
    st.dataframe(table_print)


    ## second part - the plot
    st.markdown('If we see the plot of previous data above:')
    
    ## save list of the count values and percentage value for annotate later
    count_values = table_print['count'].tolist()
    percentage_values = table_print['percentage'].tolist()

    fig = plt.figure(figsize = (11, 8)) # set figure size

    ## looping for data label
    for i in range(len(count_values)):
        
        # hard-coded coordinate text for each i
        if i == 0: # for 'no'
            xy = (i, count_values[i]//2)
        else: # for 'yes'
            xy = (i, count_values[i]//3)

        plt.annotate(
            text = percentage_values[i], # show percentage on text
            xy = xy, # coordinate text as above specified
            xytext = (0, 5),
            textcoords = 'offset points',
            ha = 'center'
        )

    sns.countplot(x='y', data=df, 
                palette=['steelblue', 'darkorange'], # to differentiate colors
                hue='y', # to prevent seaborn warning for using palette
                legend=False) # to prevent seaborn warning for using palette but not as a legend

    plt.xlabel('Customer subscription status') # set x label
    plt.ylabel('Frequencies')
    plt.title('Percentage of Subscribed vs Non-Subscribed Customers', fontweight='bold', fontsize=16, y=1)

    plt.tight_layout()
    st.pyplot(fig) # show the plot


    # insight after plot
    st.markdown(
        '''
        By the plot above we can see that from this historical campaign calls dataset, most values are customer 
        who ended up not subscribed a term deposit products. There are 88.3 % from 45195 records (after duplicates data removed), 
        which **39906 customers** that ended up as non-subscribed customers. On other hand there are 11.7 % from 45195 customers, 
        which **5289 customers** that ended up as a **subscriber** of a term deposit from this dataset. 

        From this information we know that our target variable `y` is imbalanced, so we cannot use **accuracy** as metric evaluation 
        later on modeling process. We can also note that from this information, the goal for this project is by when later use the 
        model prediction, the marketing resources can be more effective. Because after use the prediction model, the targeted customer 
        that will be called are more potential customers that likely to subscribe, rather than using much resources by reach all 
        customers without knowing potential prediction.
        '''
    )


# for EDA question 2
def EDA_q2(df):
    st.write('### 2. How is the distribution of subscribed vs non-subscribed customers for each marital status type ?')
    st.markdown(
        '''
        To answer this question, we must group the data by `marital` and `y` then calculate each frequencies:
        '''
    )

    ## first part - the dataframe of frequencies
    ## assign to the new dataframe
    marital_df = df.groupby(['marital', 'y'])['y'].count().reset_index(name='frequencies')

    st.dataframe(marital_df) # show the dataframe

    
    ## second part - the plot
    st.markdown('From the table dataframe above, we can see on the plot as below:')

    ## save the count values for annotate later
    count_values = marital_df['frequencies'].tolist()

    ## set x_positions hardcoded
    ## width bar 0.4 and space from each marital 0.6
    x_positions = [-0.2, 0.2, # divorced
                0.8, 1.2, # married
                1.8, 2.2] # single

    fig = plt.figure(figsize = (14, 8)) # set figure size

    ## looping for data label
    for i in range(len(count_values)):
        
        # hard-coded coordinate text for each i
        xy = (x_positions[i], count_values[i]) # show on top

        plt.annotate(
            text = count_values[i], # show frequencies on text
            xy = xy, # coordinate text as above specified
            xytext = (0, 5),
            textcoords = 'offset points',
            ha = 'center'
        )

    ## seaborn bar plot
    sns.barplot(data = marital_df, x = 'marital', y = 'frequencies',
                palette=['steelblue', 'darkorange'], # to differentiate colors
                hue='y'), # to prevent seaborn warning for using palette

    plt.xlabel('Customer marital status', y = 1) # set x label
    plt.ylabel('Frequencies')
    plt.title('Frequencies of Subscribed vs Non-Subscribed Customers for Each Marital Type', fontweight='bold', fontsize=16, y=1)

    plt.tight_layout()
    st.pyplot(fig) # show the plot


    # insight after plot
    st.markdown(
        '''
        Based on the plot above, we can observe the distributions of subscribed and non-subscribed customers for each marital status. 
        There are **24453 customers** with **married** marital status who ended up **not subscribed** to a term deposit products, 
        from total 27208 customers (by adding 24453 and 2755). For the other marital type, there are **10868 customers** with **single** 
        marital status which ended up **not subscribed** to a term deposit products from total 12780 customers (by adding 10868 and 1912). 
        Lastly, there are **4585 customers** with **divorced** marital status who ended up **not subscribed** to a term deposit products 
        from total 5207 customers (by adding 4585 and 622).

        The patterns for each marital type are similar with previous overall frequency distribution of subscription status, where there are 
        more customers that ended up non-subscribed (`y = no`) compare with subscribed. However, from the bar plot above among all three 
        marital categories, married customers which the largest group in terms total data also by the **non-subscribed customers ratio**. 
        This indicates that the `marital` show some relationship with the target variable `y`, that we will need check further on modeling 
        process later on **Feature Selection** section. We can also note that, the gap between subscribed and non-subscribed relatively large, 
        which suggests **`marital` alone** may **not a strong feature to be a predictor** variable. Therefore, this feature should be evaluate 
        further with combination other features and be checked in **Feature Selection** section.
        '''
    )


# for EDA question 3
def EDA_q3(df):
    st.write('### 3. Highest frequencies of clients contacted date of the year')
    st.markdown(
        '''
        To answer this question, we need to concatenate `day` and `month` as date 
        of the year (which we do not know what year from the data source) 
        as a new column. This column will not be used for future modeling, 
        only to visualize and answer EDA question. 
        
        For example, we use year 1900 just to visualize what date and month 
        of the year that clients most contacted from historical data. After that, 
        we plot the date and each frequencies of contacted clients:
        '''
    )

    ## create separate data only use for EDA question 1
    df_1 = df.copy()[['day', 'month']] # only these 2 columns

    ## for example we use year 1900
    ## this will not use to the model because we do not know exact year for each records
    year = 1900

    ## map the month to number
    month_number = {
        'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4,
        'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8,
        'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12
    }

    ## create new column month that filled with number
    df_1['month_num'] = df_1['month'].map(month_number)

    ## create date column
    df_1['date'] = pd.to_datetime({'year' : year, 
                                'month' : df_1['month_num'], 
                                'day' : df_1['day']})

    ## create series of frequencies
    date_frequencies = pd.DataFrame([
        df_1['date'].value_counts().sort_index() # the date sorted ascending
        ]).T # transpose
    
    ## plot for EDA question 1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date_frequencies.index, date_frequencies['count'], linestyle='-', color='b')
    ax.set_title('Distribution of Number of Customers Contacted', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)

    ## insight from plot above
    st.markdown(
        '''
        Earlier we knew that year 1900 above solely used to visualize the frequencies of 
        each date and month of the year (so the year can be different for each records in actual). 
        By above plot we have information that customers more frequently contacted on month May 
        and June (not specific on the same year, but we know the patterns), reached more than 1000 
        calls. While on other months usually on the level 500 - 600 calls, but much less contacted 
        on March, September, October, and December that only less than 100 calls. However, it is also 
        noticeable that there is sudden spike on November, that in total (for all years from original data) 
        there are more than 800 calls.

        From the exploratory analysis above, we observed the campaign activities most frequent on 
        May and June. Although this pseudo temporal visualization does not represent actual 
        chronological time, because may the data are collected from different year, it provides useful 
        insights into marketing operational behaviour. Since the data originally not include reliable 
        year-level information and the order of observations not informed as time progression, this pattern 
        will not be directly incorporated into the predictive model. Hence, we cannot predict with 
        **time series model** in this case, the `day` and `month` features will be treated as 
        **categorical variables** for **supervised classification** model framework instead. In the next 
        section, we will proceed to the data preprocessing and modeling phase, along with other features 
        from original dataset.
        '''
    )


# for EDA question 4
def EDA_q4(df):
    st.write('### 4. Does past campaign informations (`pdays`, `previous`, and `poutcome`) have strong association with target ?')

    ## first part
    st.markdown(
        '''
        First, if we see the data frequencies if `pdays = -1`, that from the dataset source means 
        the client was not previously contacted (previous of the current historical data timeline):
        '''
    )
    pdays_frequencies = df[
        (df['pdays'] == -1)
    ][['previous', 'poutcome']].value_counts()
    st.dataframe(pdays_frequencies)

    ## second part
    st.markdown(
        '''
        Customers with `pdays = -1` consistent have no previous contact, so the number of this 
        customer contacted is 0 (`previous`) and the outcome is unknown (`poutcome`) because no 
        outcome. For all of these customer, mean there is no previous information of the customer 
        from historical dataset (that means no previous historic data, previous of historical 
        data timeline on the dataset).

        If we see from 36938 customers that previously have not contacted (previous of the 
        historical dataset timeline):
        '''
    )
    pdays_y_frequencies = df[
        (df['pdays'] == -1)
    ][['previous', 'poutcome', 'y']].value_counts()
    st.dataframe(pdays_y_frequencies)

    ## third part
    st.markdown(
        '''
        Above output shows from 36938 customers that previously not contacted, the result 
        where customer subscribed only 3384 out of 36938 customers. This mean from historical 
        campaign record, the percentage rate of new customer approached and successfully 
        subscribed is about 9%. By development of this machine learning model to predict customer 
        will subscribe by this past historical data, the hope is to increase this percentage.
        '''
    )
    st.markdown(
        '''
        Next, if we see the association of each of these 3 features with target variable `y`:
        > Note that, in this **EDA** section we use all dataset `df`. However on **Feature Selection** 
        when modeling process we obviously will use train-set only
        '''
    )

    ### compute Phi-K correlation for each set of columns and print
    correlation = compute_phik_correlation(
        df[['pdays', 'previous', 'poutcome', 'y']],
        df[['pdays', 'previous', 'poutcome', 'y']].columns.to_list(),
        target=['y']
    )

    ## show correlation result
    st.write('#### Phi-K Correlation with Target (y)')
    st.dataframe(correlation) # sort ascending

    ## insight from correlation result
    st.markdown(
        '''
        > **References** to use Phi-K correlation value:
        > - [Baak, M., Koopman, R., Snoek, H., & Klous, S. (2019)](https://arxiv.org/pdf/1811.11440)
        > - [phik-documentation](https://phik.readthedocs.io/_/downloads/en/stable/pdf/)

        By above output, Phi-K correlation value shows that `poutcome` has the strongest association with 
        the target, from these 3 features of past campaign information (past of current historical data timeline). 
        We can also see that `pdays` have Phi-K correlation value 0.25, lower than `poutcome` but not so low. 
        However, it is noticeable that `previous` have the lowest and low value of Phi-K correlation value 
        that almost 0, indicates **no association** with the target variable.

        Although the above result, we check information of all of the dataset `df` just for **EDA** purposes. 
        The real association check of all features will be done on **Feature Selection** section when modeling, 
        with only train data. The purpose is to avoid information leakage and ensure feature selection based 
        solely on training data, and the test data follow behavior of **unseen data** and as correct benchmark 
        for model evaluation.
        '''
    )


# for EDA question 5
def EDA_q5(df):
    st.write('### 5. Distribution of numerical features (`age`, `balance`, `campaign`, `pdays`, and `previous`)')
    st.write('If we see the distribution plot of each numerical features:')

    ## first distribution plot
    num_cols = ['age', 'balance', 'campaign', 'pdays', 'previous']

    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 12)) # create 2x3 subplots

    ### initial value for axes
    nrow = 0
    ncol = 0

    for col in num_cols:

        # set axes
        ax = axes[nrow, ncol]

        # histogram plot to see distribution
        # use df since on EDA section
        sns.histplot(df[col], bins = 40, ax = ax)
        ax.set_xlabel(col)

        ncol += 1 # increase the ncol

        # ncol should only 0, 1, or 2
        # if after increment earlier reached 3
        if ncol == 3: 
            ncol = 0 # reset to 0
            nrow += 1 # next row

    ### turned off unused axis
    axes[1, 2].axis('off')

    ### display the plot and styling
    fig.suptitle('Distribution of Numerical Features', fontsize = 16, fontweight = 'bold', y = 1)
    plt.tight_layout()
    st.pyplot(fig)

    ## insight from first distribution plot
    st.markdown(
        '''
        It is noticeable that all of the numerical variable above have **positive skew** distribution based 
        on historical dataset. Only column `age` that the distribution *may* look not extremely skew, but 
        the other variables *indicates* have **extreme positive skew distribution**. Regarding normality, 
        if we check further with D'Agostino K-Squared Test to check normality:
        > **H0: the data follows normal distribution**   
        > **H1: the data does not follows normal distribution**
        '''
    )

    ## runner skewness and normality test
    skew_list = []
    pvalue_list = []

    for col in num_cols:
        # skewness value
        skew = df[col].skew()

        # d'agostino statistical hypothesis test
        pvalue = stats.normaltest(df[col]).pvalue
        
        # st.write(f'For column {col}:')
        skew_list.append(skew)
        pvalue_list.append(pvalue)
    
    table_print = pd.DataFrame({
        'Skewness value' : skew_list,
        'P-value' : pvalue_list
    })
    table_print.index = num_cols # change index
    st.dataframe(table_print)

    ## insight after normality test
    st.markdown(
        '''
        Since all p-value < 0.05, we reject null hypothesis for all numerical columns earlier. That means 
        all numerical features are **not normally distributed** on this historical dataset. Based on the 
        skewness value, only column `age` that is **moderate positive skew**, the other are **extreme 
        positive skew**. Although on the modeling process later we need to based on **training set only**, 
        this behavior should be follows since train-test randomly split. The process to see distribution of 
        all numerical dataset based on **full historical dataset** is complete. We only done for **EDA purposes**, 
        and will do re-checking for modeling purposes later with only check the distribution on the training set.
        '''
    )


# for EDA question 6
def EDA_q6(df):
    st.write('### 6. Are there numerical features that need logarithmic transformation ?')
    st.markdown(
        '''
        Note that the **extreme positive** skew distribution earlier are too extremely skew, there will be too many 
        data resided as **outliers** later. We may apply re-shaping to handle this by **natural logarithmic transformation**. 
        We do not need to apply to `age` since the distribution plot earlier already have the shape. We also cannot apply 
        to the variable `balance` because earlier we known that there are many negative values (consider customers who currently 
        in the state of debt). Hence, for the other three numerical variables, if we see the result:
        > We will do log-transformation procedure of reshaping process of numerical variable for later in the section 
        **Outliers Handling**, one of transformation technique that effective to normalize numerical variable. References:   
        > - [Kaggle: logarithm-is-effective-at-normalizing](https://www.kaggle.com/code/ryanholbrook/creating-features)   
        > - [Benefit-closeness-to-normality](https://towardsdatascience.com/logarithmic-transformation-for-beginners-99488b8951e3/)
        '''
    )

    ## distribution after log-transformed
    log1p_cols = ['campaign', 'previous']
    log2p_cols = ['pdays'] # since pdays there is -1 values, need log(2+x) instead of log(1+x)

    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (14, 18)) # create 3x2 subplots

    log_cols = log1p_cols + log2p_cols

    for i in range(len(log_cols)):
        col = log_cols[i]

        # left plot - original
        sns.histplot(df[col], bins = 40, ax = axes[i, 0])
        axes[i, 0].set_xlabel(col)

        # right plot - log-transformed
        # log(2+x) for pdays (not in log1p_cols)
        sns.histplot(np.log1p(df[col]) if col in log1p_cols else np.log1p(df[col] + 1), 
                    bins = 40, ax = axes[i, 1])
        axes[i, 1].set_xlabel(
            f'{col} (Log(1+x)-Transformed)' if col in log1p_cols else f'{col} (Log(2+x)-Transformed)'
        )

    ### display the plot and styling
    fig.suptitle(f'Distribution of {log_cols} - Before and After Log Transformed', fontsize = 16, weight = 'bold', y = 1)
    plt.tight_layout()
    st.pyplot(fig)

    ## insight after distribution log-transformed
    st.markdown(
        '''
        By above result, after re-shaping log-transformed log(1+x) (except for `pdays` with log(2+x) since there are many -1 values) 
        the distributions are more stable and outliers more handled. However, if we see further on previous plot, even after log-transformation, 
        the distribution more focus on 1 part that is **0** after log-transformation (before log-transformation, `pdays` focus on -1 value, 
        while `previous` focus on 0 value). That means both of these columns **zero-inflated** columns.
        '''
    )

    ## zero-inflated percentage print
    zero_inflated = ['pdays', 'previous']
    top_most_values = []
    percentage_top = []

    for col in zero_inflated:
        top_most_values.append(df[col].value_counts().index[0]) # top most values
        
        top_freq = df[col].value_counts().iloc[0] # top most frequencies
        percentage = top_freq * 100 / df.shape[0]

        percentage_top.append(f'{percentage:.2f} %') 

    table_print = pd.DataFrame({
        'Top most value' : top_most_values,
        'Percentage in data' : percentage_top
    })
    table_print.index = zero_inflated # change index
    st.dataframe(table_print)
        
    ## insight regarding zero-inflated features of past campaigns
    st.markdown(
        '''
        By above result we have information there **inflated** focus to 1 values for both these numerical columns `pdays` and `previous`. 
        From earlier information on **EDA** section 2, we have information that both of these columns are explain past campaign information. 
        Later we will try to re-apply this process in **Outliers Handling** section, and explore further to see what to do in modeling process. 
        For current exploratory data analysis in this section, we found that the logarithmic transformation made distribution of `campaign` more 
        stable and can be used for this variable with further handling on **Outliers Handling** section, **but for `pdays` and `previous`** need 
        **further inspection** regarding **zero-inflated** behavior.
        '''
    )


# for run all in the function below
def run():

    # create title
    st.title('Data-Driven Term Deposit Subscription Modeling for Marketing Campaigns')
    st.write('## Exploratory Data Analysis (EDA)')
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

    # line to separate
    st.markdown('---')

    # variable descriptions
    st.markdown(
        """
        Variable Descriptions:

        | Column | Description |
        | --- | --- |
        | `age` | age of the client |
        | `job` | type of job of the client |
        | `marital` | marital status |
        | `education` | education level |
        | `default` | the client has credit in default ? |
        | `balance` | average yearly balance |
        | `housing` | the client has housing loan ? |
        | `loan` | the client has personal loan ? |
        | `contact` | contact communication type |
        | `day` | last contact day of the month |
        | `month` | last contact month of the year |
        | `duration` | the duration in second of this call data |
        | `campaign` | number of contacts performed during this campaign and for this client |
        | `pdays` | number of days that passed by after the client was last contacted from a previous campaign 
        (`-1` means the client was not previously contacted) |
        | `previous` | number of contacts performed before this campaign and for this client |
        | `poutcome` | outcome of the previous marketing campaign |
        | `y` | has the client subscribed a term deposit ? |


        > **Special Note:**  
        > From the dataset source informaton, the column `duration` highly affects the output target, 
        since in realistic scenario when the call is performed and have a duration of the call, it is obviously 
        known what the result (`y`) at the end of the call. So, this column should be discarded if the intention to 
        be applicable for realistic scenario
        """
    )

    # line to separate
    st.markdown('---')


    # load the data before EDA
    df = pd.read_csv('./src/bank-full.csv', sep = ';')
    df['day'] = df['day'].astype(object) # change data type day
    df = df.drop('duration', axis = 1) # drop column `duration`
    df = df.drop_duplicates().reset_index(drop=True) # drop duplicates

    
    # subtitle
    st.write('## EDA Visualizations')
    st.markdown(
        '''
        Main question for **Exploratory Data Analysis (EDA)** of this project, arise from the questions:

        1. What is the percentage of subscribed vs. non-subscribed customers in this historical dataset?   
        2. How is the distribution of subscribed vs non-subscribed customers for each marital status type?   
        3. When is the date of the year that have the highest frequencies of clients contacted? Are there specific 
        months of the year when past clients are contacted more frequently?   
        4. Does past campaign informations, which is `pdays`, `previous`, and `poutcome` features have strong association 
        to predict variable whether the client will subscribe term deposit?   
        5. How is the distribution of numerical features (`age`, `balance`, `campaign`, `pdays`, and `previous`)? 
        Does all have normal distribution?   
        6. From the answer of the question no.4, are there numerical features that need logarithmic transformation?
        '''
    )

    ## EDA question 1
    EDA_q1(df)
    st.markdown('---')

    ## EDA question 2
    EDA_q2(df)
    st.markdown('---')

    ## EDA question 3
    EDA_q3(df)
    st.markdown('---')

    ## EDA question 4
    EDA_q4(df)
    st.markdown('---')

    ## EDA question 5
    EDA_q5(df)
    st.markdown('---')

    ## EDA question 6
    EDA_q6(df)


if __name__=='__main__':
    run()