import spacy
import en_core_web_sm
import streamlit as st

# Import necessary libraries
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Import python files we've created to help
from nlp import *
from data_cleaning import *
from classification import *
from visualizations import *


# Load the small English model
nlp = en_core_web_sm.load()

st.title('Customer Conversation Classifier')
st.subheader('Learn how your customers feel about your products')

txt = st.text_area('Paste text from customer here (product review, tweet, etc):', '')

if (st.button('Let me know!')): 
    score = sentiment_analyzer_scores(txt)

    
    clf_forest =  pickle.load( open( "../models/save.forest", "rb" ))

    X_txt = pd.DataFrame({'review_fulltext': [txt]})
    X_txt = append_sentiment_scores(X_txt)


    # if txt.strip() == '': 
    if (X_txt['compound'][0] >= -.05) & (X_txt['compound'][0] <= .05):
        y_pred = 0.5
        msg = "Very neutral customer comment. I really can't call it."
    else: 
        y_pred = clf_forest.predict(X_txt.drop(columns='review_fulltext'))[0]
        if y_pred == 0: msg = "This customer may not be the happiest."
        else: msg = 'This customer seems pleased. Congratulations!'    

    st.header(sentiment_emoji(y_pred))
    st.subheader(msg)

    f, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(score.keys()))
    ax.set_title('Review Sentiment')
    bar = ax.barh(y_pos, list(score.values()), color=SEABORN_PALETTE[1])
    bar[3].set_color(SEABORN_PALETTE[4])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(['Negative', 'Neutral', 'Positive', 'Overall'])
    sns.despine()
    st.pyplot()

#     doc = nlp(txt)
    
#     # st.write(spacy.displacy.render(doc, style='ent', jupyter=True))
    
#     # Iterate over the tokens
#     for token in doc:
#         # Print the text and the predicted part-of-speech tag
#         st.write(token.text, token.pos_, token.dep_, token.head.text)
