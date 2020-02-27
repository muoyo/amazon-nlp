"""
##### NATURAL LANGUAGE PROCESSING #####

This module contains functions for dealing with Natural Language Processing (NLP)

"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


analyzer = SentimentIntensityAnalyzer()

# Return dictionary of VADER sentiment scores for given sentence. Keys: 'neg', 'neu', 'pos', 'compound'
def sentiment_analyzer_scores(sentence, verbose=False):
    score = analyzer.polarity_scores(sentence)
    if verbose: print("{}".format(str(score)))
    
    return score
    
def sentiment_emoji(compound_score):
    if compound_score == 1: return '😊'
    if compound_score == 0: return '😡'
    return '😐'
    
def sentiment_analyzer_scores_neg(sentence):
    return sentiment_analyzer_scores(sentence)['neg']

def sentiment_analyzer_scores_neu(sentence):
    return sentiment_analyzer_scores(sentence)['neu']

def sentiment_analyzer_scores_pos(sentence):
    return sentiment_analyzer_scores(sentence)['pos']

def sentiment_analyzer_scores_compound(sentence):
    return sentiment_analyzer_scores(sentence)['compound']

def append_sentiment_scores(df):
    df['neg'] = df['review_fulltext'].apply(sentiment_analyzer_scores_neg)
    df['neu'] = df['review_fulltext'].apply(sentiment_analyzer_scores_neu)
    df['pos'] = df['review_fulltext'].apply(sentiment_analyzer_scores_pos)
    df['compound'] = df['review_fulltext'].apply(sentiment_analyzer_scores_compound)
        
    return df

def get_vectorized_features(X_train, X_test):
    word_tokenizer = RegexpTokenizer(r'\w+')
    
    X_train = X_train.loc[:, 'review_fulltext']
    X_test = X_test.loc[:, 'review_fulltext']
    
    #split each row into a list and remove punctuation. 
    X_train = X_train.map(lambda x: word_tokenizer.tokenize(x.lower()))
    X_test = X_test.map(lambda x: word_tokenizer.tokenize(x.lower()))
    
    # rejoin list of tokenized words into single string for each row
    X_train = X_train.map(lambda x: ' '.join(x))
    X_test = X_test.map(lambda x: ' '.join(x))
    
    clean_train_data = []
    clean_test_data = []

    for traindata in X_train:
        clean_train_data.append(traindata)

    for testdata in X_test:
        clean_test_data.append(testdata)
        
    # instantiate our CountVectorizer. This counts the number of appearances of all the words in our training data and
    # eliminates common english stop words. 5000 max features works well for our purposes (tested various numbers). Our
    # data is already preprocessed and tokenized manually earlier. ngram_range is 1,3, although all or nearly all our 
    # features are single words

    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = 'english',
                                 max_features = 5000,
                                 ngram_range = (1, 3))
    
    # fit our training data and test data lists to our count_vectorizer
    train_data_features = vectorizer.fit_transform(clean_train_data)
    test_data_features = vectorizer.transform(clean_test_data)
    
    # convert to array
    train_data_features = train_data_features.toarray()
    
    return train_data_features, test_data_features