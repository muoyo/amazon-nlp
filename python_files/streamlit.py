import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

# Import necessary libraries
import pickle
import warnings
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# warnings.filterwarnings('ignore')

# Import python files we've created to help
import subprocess

from nlp import *
from data_cleaning import *
from classification import *
from visualizations import *


st.title('Customer Conversation Classifier')
st.subheader('Do your customers love you, or nah?')

txt = st.text_area('Paste an Amazon review here...', '', key='textarea')
score = sentiment_analyzer_scores(txt)

if (st.button('Submit')): score = sentiment_analyzer_scores(txt)

    
    
clf_forest =  pickle.load( open( "../models/save.forest", "rb" ))

X_txt = pd.DataFrame({'review_fulltext': [txt]})
X_txt = append_sentiment_scores(X_txt)


if txt.strip() == '': 
    y_pred = 0.5
    msg = "Very neutral customer comment. I really can't call it."
else: 
    y_pred = clf_forest.predict(X_txt.drop(columns='review_fulltext'))[0]
    if y_pred == 0: msg = "This customer may not be the happiest."
    else:  msg = 'This customer seems pleased. Congratulations!'
st.header(sentiment_emoji(y_pred))
st.subheader(msg)

st.write(score)
X_txt
    



# Read in original data
# df_full = pd.read_csv('https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Watches_v1_00.tsv.gz', sep='\t', error_bad_lines=False, warn_bad_lines=False)

# Clean data & load into final dataframe
# df_full = df_full[df_full.star_rating != 3]
# df_full['review_class'] = ((df_full['star_rating'] == 4) | (df_full['star_rating'] == 5)).astype(int)

# df = clean_data(df_full, 25000)


# df_full.customer_id.value_counts()

# df = append_sentiment_scores(df)

# df = pickle.load( open( "../notebooks/save.df", "rb" ) )

# X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=.25)
# y_train.value_counts()


# X_train_numeric = X_train.select_dtypes(include=[np.number])
# X_test_numeric = X_test.select_dtypes(include=[np.number])


# from imblearn.under_sampling import RandomUnderSampler

# rus = RandomUnderSampler()
# X_train_resampled, y_train_resampled = rus.fit_resample(X_train_numeric, y_train)

    
    


# # Logistic Regression
# clf_lr = LogisticRegression(fit_intercept=True, C=1e12, solver='liblinear', penalty='l2')
# clf_lr.fit(X_train_resampled, y_train_resampled)

# y_hat_lr_train = clf_lr.predict(X_train_resampled)
# st.write(classification_report(y_train_resampled, y_hat_lr_train))
# show_confusion_matrix(clf_lr, X_train_resampled, y_train_resampled, title='Logistic Regression - Training Set (Normalized)')
# st.pyplot()

# y_hat_lr_test = clf_lr.predict(X_test_numeric)
# st.write(classification_report(y_test, y_hat_lr_test))
# show_confusion_matrix(clf_lr, X_test_numeric, y_test, title='Logistic Regression - Test Set (Normalized)')
# st.pyplot()


# # K Nearest Neighbors
# clf_knn = KNeighborsClassifier()
# clf_knn.fit(X_train_resampled, y_train_resampled)

# y_hat_knn_train = clf_knn.predict(X_train_resampled)
# st.write(classification_report(y_train_resampled, y_hat_knn_train))
# show_confusion_matrix(clf_knn, X_train_resampled, y_train_resampled, title='K Nearest Neighbors - Training Set (Normalized)')
# st.pyplot()

# y_hat_knn_test = clf_knn.predict(X_test_numeric)
# st.write(classification_report(y_test, y_hat_knn_test))
# show_confusion_matrix(clf_knn, X_test_numeric, y_test, title='K Nearest Neighbors - Test Set (Normalized)')
# st.pyplot()


# # Decision Trees
# clf_dt = DecisionTreeClassifier(criterion='entropy')
# clf_dt.fit(X_train_resampled, y_train_resampled)

# y_hat_dt_train = clf_dt.predict(X_train_resampled)
# st.write(classification_report(y_train_resampled, y_hat_dt_train))
# show_confusion_matrix(clf_dt, X_train_resampled, y_train_resampled, title='Decision Tree - Training Set (Normalized)')
# st.pyplot()

# y_hat_dt_test = clf_dt.predict(X_test_numeric)
# st.write(classification_report(y_test, y_hat_dt_test))
# show_confusion_matrix(clf_dt, X_test_numeric, y_test, title='Decision Tree - Test Set (Normalized)')
# st.pyplot()


# # Bagged Trees
# clf_bagged = BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_leaf_nodes=100), 
#                                  n_estimators=20)
# clf_bagged.fit(X_train_resampled, y_train_resampled)

# y_hat_bagged_train = clf_bagged.predict(X_train_resampled)
# st.write(classification_report(y_train_resampled, y_hat_bagged_train))
# show_confusion_matrix(clf_bagged, X_train_resampled, y_train_resampled, title='Bagged Trees - Training Set (Normalized)')
# st.pyplot()

# y_hat_bagged_test = clf_bagged.predict(X_test_numeric)
# st.write(classification_report(y_test, y_hat_bagged_test))
# show_confusion_matrix(clf_bagged, X_test_numeric, y_test, title='Bagged Trees - Test Set (Normalized)')
# st.pyplot()


# # Random Forest
# clf_forest = RandomForestClassifier(n_estimators=100, max_depth = 15)
# clf_forest.fit(X_train_resampled, y_train_resampled)

# y_hat_forest_train = clf_forest.predict(X_train_resampled)
# st.write(classification_report(y_train_resampled, y_hat_forest_train))
# show_confusion_matrix(clf_forest, X_train_resampled, y_train_resampled, title='Random Forest - Training Set (Normalized)')
# st.pyplot()

# # y_hat_forest_test = clf_forest.predict(X_test_numeric)
# # st.write(classification_report(y_test, y_hat_forest_test))
# show_confusion_matrix(clf_forest, X_test_numeric, y_test, title='Random Forest - Test Set (Normalized)')
# st.pyplot()


# # Adaboost
# clf_ab = AdaBoostClassifier()
# clf_ab.fit(X_train_resampled, y_train_resampled)

# y_hat_ab_train = clf_ab.predict(X_train_resampled)
# st.write(classification_report(y_train_resampled, y_hat_ab_train))
# show_confusion_matrix(clf_ab, X_train_resampled, y_train_resampled, title='Adaboost - Training Set (Normalized)')
# st.pyplot()

# y_hat_ab_test = clf_ab.predict(X_test_numeric)
# st.write(classification_report(y_test, y_hat_ab_test))
# show_confusion_matrix(clf_ab, X_test_numeric, y_test, title='Adaboost - Test Set (Normalized)')
# st.pyplot()


# # Gradient Boost
# clf_gb = GradientBoostingClassifier()
# clf_gb.fit(X_train_resampled, y_train_resampled)

# y_hat_gb_train = clf_gb.predict(X_train_resampled)

# st.write(classification_report(y_train_resampled, y_hat_gb_train))
# show_confusion_matrix(clf_gb, X_train_resampled, y_train_resampled, title='Gradient Boost - Training Set (Normalized)')
# st.pyplot()

# y_hat_gb_test = clf_gb.predict(X_test_numeric)
# st.write(classification_report(y_test, y_hat_gb_test))
# show_confusion_matrix(clf_gb, X_test_numeric, y_test, title='Gradient Boost - Test Set (Normalized)')
# st.pyplot()


# DATE_TIME = "date/time"
# DATA_URL = (
#     "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
# )

# st.title("Uber Pickups in New York City")
# st.markdown(
# """
# This is a demo of a Streamlit app that shows the Uber pickups
# geographical distribution in New York City. Use the slider
# to pick a specific hour and look at how the charts change.
# [See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
# """)

# @st.cache(persist=True)
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis="columns", inplace=True)
#     data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
#     return data


# data = load_data(100000)



# hour = st.sidebar.slider('hour', 0, 23, 10)
# data = data[data[DATE_TIME].dt.hour == hour]

# if st.sidebar.checkbox('Show Raw Data'):
    
#     'data', data

    
# midpoint = (np.average(data["lat"]), np.average(data["lon"]))

# st.write(pdk.Deck(
#     map_style="mapbox://styles/mapbox/light-v9",
#     initial_view_state={
#         "latitude": midpoint[0],
#         "longitude": midpoint[1],
#         "zoom": 11,
#         "pitch": 50,
#     },
#     layers=[
#         pdk.Layer(
#             "HexagonLayer",
#             data=data,
#             get_position=["lon", "lat"],
#             radius=100,
#             elevation_scale=4,
#             elevation_range=[0, 1000],
#             pickable=True,
#             extruded=True,
#         ),
#     ],
# ))

