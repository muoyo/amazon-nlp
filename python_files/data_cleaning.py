"""
##### DATA CLEANING #####

This module is for data cleaning.

"""



def clean_data(df, numrows=False):
    """
    This function runs our support functions to clean the data before returning a final dataframe for analysis
    
    :return: cleaned dataset to be passed to other modules.
    """    
    
    if numrows: df = df.sample(n=numrows)

    # Deal with missing values for any of the columns we will be using
    df['review_headline'] = df['review_headline'].fillna('.')
    df['review_body'] = df['review_body'].fillna('.')
    df['review_fulltext'] = df['review_headline'] + '. ' + df['review_body']
    
    return df
