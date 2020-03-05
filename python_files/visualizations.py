"""
##### DATA VISUALIZATIONS #####

This module contains the functions for all the visualizations for our project.

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import plot_confusion_matrix


# Controls appearance of seaborn plots. Options: paper, notebook, talk, or poster
SEABORN_CONTEXT = 'notebook' 
SEABORN_PALETTE = sns.color_palette("bright")
sns.set_context(SEABORN_CONTEXT)

local_stopwords = ['watch', 'watches', 'watche', 'br', 'five', 'star', 'good', 'old', 'year', 'way', 'say', 
                   'great', 'happy', 'nice', 'stars', 'love', 'used', 'problem', 'second', 'highly', 'recommend',
                   'time', 'one', 'two', 'three', 'four', 'look', 'band', 'Amazon']

def wordcloud(wordstring, stopwords=False, figsize=(20,20)):
    sw = set(STOPWORDS)
    sw.update(local_stopwords)
    
    if stopwords: sw.update(stopwords)
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud(stopwords=sw).generate(wordstring)
    
    # Display the generated image:
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    
def class_c_review_wordcloud (df, c, stopwords=False, figsize=(20,20)):
    class_c_reviews = df.loc[df['review_class'] == c]['review_fulltext']
    class_c_reviews = class_c_reviews.str.cat(sep=' ')
    
    wordcloud(class_c_reviews, stopwords, figsize)


def n_star_review_wordcloud (df, n, stopwords=False, figsize=(20,20)):
    n_star_reviews = df.loc[df['star_rating'] == n]['review_fulltext']
    n_star_reviews = n_star_reviews.str.cat(sep=' ')
    
    wordcloud(n_star_reviews, stopwords, figsize)


def show_confusion_matrix(classifier, X, y, title='', figsize=(10,10), cmap=plt.cm.Blues, normalize='true'):
    f, ax = plt.subplots(figsize=figsize)
    cm = plot_confusion_matrix(classifier, X, y, cmap=cmap, ax=ax, normalize=normalize)
    ax.set_title(title);
    
    return cm
    
    
def barplot(group_series, title='', xlabel='', ylabel='', figsize=(15,10), color=SEABORN_PALETTE[2]):

    
    fig, ax = plt.subplots(figsize=figsize)
    group_series.plot.bar(color=color, label=ylabel, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);
    plot_mean_and_ci(group_series, ax, color)
    ax.legend()
    
    return ax



def plot_mean_and_ci(y_series, ax, color):    
    ax.axhline(y_series.mean(), color=color, linewidth=5, linestyle='-', label='mean')
    ax.axhspan(y_series.quantile(.05), y_series.quantile(.95), color=color, alpha=0.25, label='95% confidence')

    return ax
