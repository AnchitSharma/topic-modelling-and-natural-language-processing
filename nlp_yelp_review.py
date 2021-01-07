import pickle
import gensim
import pyLDAvis
import pyLDAvis.gensim
import spacy
import pandas as pd
import nltk; nltk.download('stopwords')
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import re
import warnings
from pprint import pprint
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
#%config InlineBackend.figure_formats = ['retina']
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

with open(r'D:\anchit_workspace\nlp\nlp_yelp_review_unsupervised-master\nlp_yelp_review_unsupervised-master\data\rev_train.pkl', 'rb') as f:
    rev_train = pickle.load(f)
with open(r'D:\anchit_workspace\nlp\nlp_yelp_review_unsupervised-master\nlp_yelp_review_unsupervised-master\data\rev_test.pkl', 'rb') as f:
    rev_test = pickle.load(f)
    
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['come','order','try','go','get','make','drink','plate','dish','restaurant','place','would','really','like','great','service','came','got']) 

def strip_newline(series):
    return [review.replace('\n','') for review in series]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
# =============================================================================
# def remove_stopwords(texts):
#     return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
# 
# =============================================================================
def remove_stopwords(texts):
    out = [[word for word in simple_preprocess(str(doc))
            if word not in stop_words]
            for doc in texts]
    return out
def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod
def get_corpus(df):
    df['text'] = strip_newline(df.text)
    words = list(sent_to_words(df.text))
    words = remove_stopwords(words)
    bigram_mod = bigrams(words)
    bigram = [bigram_mod[review] for review in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    
    return corpus, id2word, bigram
train_corpus, train_id2word, bigram_train = get_corpus(rev_train)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lda_train = gensim.models.ldamulticore.LdaMulticore(
                           corpus=train_corpus,
                           num_topics=20,
                           id2word=train_id2word,
                           chunksize=100,
                           workers=7, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = 1,
                           per_word_topics=True)
    lda_train.save('lda_train.model')
    
lda_train.print_topics(20,num_words=15)[:10]

train_vecs = []
for i in range(len(rev_train)):
    top_topics = (
        lda_train.get_document_topics(train_corpus[i],
                                      minimum_probability=0.0)
    )
    topic_vec = [top_topics[i][1] for i in range(20)]
    topic_vec.extend([rev_train.iloc[i].counts])
    topic_vec.extend([len(rev_train.iloc[i].text)])
    train_vecs.append(topic_vec)

X = np.array(train_vecs)
y = np.array(rev_train.sentiment)

kf = KFold(5, shuffle=True, random_state=42)
cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1 = [], [], []

for train_ind, val_ind in kf.split(X, y):
    # Assign CV IDX
    X_train, y_train = X[train_ind], y[train_ind]
    X_val, y_val = X[val_ind], y[val_ind]
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_val_scale = scaler.transform(X_val)
    
    # Logistic Regression
    lr = LogisticRegression(
            class_weight = 'balanced',
            solver = 'newton-cg',
            fit_intercept = True
            ).fit(X_train_scale, y_train)
    
    y_pred = lr.predict(X_val_scale)
    cv_lr_f1.append(f1_score(y_val, y_pred, average = 'binary'))
    
    # Logistic Regression SGD
    sgd = linear_model.SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            loss='log',
            class_weight='balanced',
            
            ).fit(X_train_scale, y_train)
    y_pred = sgd.predict(X_val_scale)
    cv_lrsgd_f1.append(f1_score(y_val, y_pred, average='binary'))
    
    
    # Logistic Modified huber
    sgd_huber = linear_model.SGDClassifier(
            max_iter=1000,
            tol = 1e-3,
            alpha=20,
            loss='modified_huber',
            class_weight='balanced'
            ).fit(X_train_scale, y_train)
    
    y_pred = sgd_huber.predict(X_val_scale)
    cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary'))
    

    
print(f'Logistic Regression Val f1 : {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
print(f'Logistic Regression SGD f1 : {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
print(f'SVM Huber Val f1 : {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')


# Applying the Model on unseen data
def get_bigram(df):
    df['text'] = strip_newline(df.text)
    words = list(sent_to_words(df.text))
    words = remove_stopwords(words)
    bigram = bigrams(words)
    bigram = [bigram[review] for review in words]
    return bigram


bigram_test = get_bigram(rev_test)

test_corpus = [train_id2word.doc2bow(text) for text in bigram_test]

test_vecs = []
for i in range(len(rev_test)):
    top_topics = (
            lda_train.get_document_topics(test_corpus[i], minimum_probability=0.0)
            )
    topic_vec = [top_topics[i][1] for i in range(20)]
    topic_vec.extend([rev_test.iloc[i].counts])
    topic_vec.extend([len(rev_test.iloc[i].text)])
    test_vecs.append(topic_vec)
    

ss = StandardScaler()
X= ss.fit_transform(test_vecs)
y = rev_test.sentiment

lr = LogisticRegression(
        class_weight='balanced',
        solver = 'newton-cg',
        fit_intercept=True,
        ).fit(X, y)

y_pred_lr = lr.predict(X)
print(f1_score(y, y_pred_lr, average='binary'))

sgd_huber = linear_model.SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            alpha=20,
            loss='modified_huber',
            class_weight='balanced', shuffle=True
        ).fit(X, y)

y_pred_huber = sgd_huber.predict(X)
print(f1_score(y, y_pred_huber, average='binary'))


































