import pandas as pd
import numpy as np
from IPython.core.pylabtools import figsize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

turn_diarisation_annotation_coach = pd.read_csv('./data/turn_coach_for_model.csv')

train, test = train_test_split(turn_diarisation_annotation_coach, test_size=0.2, random_state=11)
print(train.shape)
print(test.shape)
# Check if there is any NA in those transcription columns.
print(train.info())
train = train.loc[train.transcription_NoStopWords.notnull()]
train = train.loc[train.transcription_After_RemovingStopWords_lemma.notnull()]
print(train.info())
print(test.info())
test = test.loc[test.transcription_NoStopWords.notnull()]
test = test.loc[test.transcription_After_RemovingStopWords_lemma.notnull()]
print(test.info())

# Check data imbalance or not
count_False, count_True = train.label.value_counts()
print('AAAA', train.label.value_counts())
ax = train.label.value_counts().plot(kind='bar', title='Before oversampling')
plt.xlabel('class')
plt.ylabel('counts')
plt.bar_label(ax.containers[0], size=10)
#plt.show()

# Divide by class
class_False = train[train['label'] == bool(False)]
class_True = train[train['label'] == bool(True)]

# Random over-sampling
class_True_oversampling = class_True.sample(count_False, replace=True)
train = pd.concat([class_False, class_True_oversampling], axis=0)
print('Random over-sampling:')
print(train.label.value_counts())
ax = train.label.value_counts().plot(kind='bar', title='After oversampling')
plt.xlabel('class')
plt.ylabel('counts')
plt.bar_label(ax.containers[0], size=10)
#plt.show()

# Naive Bayes
def NBclassifier(train_text, train_label, test_text, test_label):
    clf = Pipeline(
        [
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ]
    )
    clf.fit(train_text, train_label)
    prediction = clf.predict(test_text)
    matrix_report = metrics.classification_report(test_label, prediction, output_dict=True)
    matrix_report = pd.DataFrame(matrix_report).transpose()
    return matrix_report

NB_transcription = NBclassifier(train['transcription'], train['label'], test['transcription'], test['label'])
NB_transcription.to_csv('./performance/coach_NB_transcription.csv', index=False)

NB_nostopwords = NBclassifier(train['transcription_NoStopWords'], train['label'], test['transcription_NoStopWords'], test['label'])
NB_nostopwords.to_csv('./performance/coach_NB_nonstopwrods.csv', index=False)

NB_lemma = NBclassifier(train['transcription_afterLemmatization'], train['label'], test['transcription_afterLemmatization'], test['label'])
NB_lemma.to_csv('./performance/coach_NB_lemma.csv', index=False)

NB_nostopwords_lemma = NBclassifier(train['transcription_After_RemovingStopWords_lemma'], train['label'], test['transcription_After_RemovingStopWords_lemma'], test['label'])
NB_nostopwords_lemma.to_csv('./performance/coach_NB_nostopwords_lemma.csv', index=False)

# Support vector machine
def SVMclassifier(train_text, train_label, test_text, test_label, regularizer):
    clf = Pipeline(
        [
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(penalty=regularizer))
        ]
    )
    clf.fit(train_text, train_label)
    prediction = clf.predict(test_text)
    matrix_report = metrics.classification_report(test_label, prediction, output_dict=True)
    matrix_report = pd.DataFrame(matrix_report).transpose()
    return matrix_report

SVM_transcription = SVMclassifier(train['transcription'], train['label'], test['transcription'], test['label'],'none')
SVM_transcription.to_csv('./performance/coach_SVM_transcription.csv', index=False)

SVM_transcription_L2 = SVMclassifier(train['transcription'], train['label'], test['transcription'], test['label'],'l2')
SVM_transcription_L2.to_csv('./performance/coach_SVM_transcription_L2.csv', index=False)

SVM_nostopwords = SVMclassifier(train['transcription_NoStopWords'], train['label'], test['transcription_NoStopWords'], test['label'], 'none')
SVM_nostopwords.to_csv('./performance/coach_SVM_nonstopwrods.csv', index=False)

SVM_nostopwords_L2 = SVMclassifier(train['transcription_NoStopWords'], train['label'], test['transcription_NoStopWords'], test['label'], 'l2')
SVM_nostopwords_L2.to_csv('./performance/coach_SVM_nonstopwrods_L2.csv', index=False)

SVM_lemma = SVMclassifier(train['transcription_afterLemmatization'], train['label'], test['transcription_afterLemmatization'], test['label'], 'none')
SVM_lemma.to_csv('./performance/coach_SVM_lemma.csv', index=False)

SVM_lemma_L2 = SVMclassifier(train['transcription_afterLemmatization'], train['label'], test['transcription_afterLemmatization'], test['label'], 'l2')
SVM_lemma_L2.to_csv('./performance/coach_SVM_lemma_L2.csv', index=False)

SVM_nostopwords_lemma = SVMclassifier(train['transcription_After_RemovingStopWords_lemma'], train['label'], test['transcription_After_RemovingStopWords_lemma'], test['label'], 'none')
SVM_nostopwords_lemma.to_csv('./performance/coach_SVM_nostopwords_lemma.csv', index=False)

SVM_nostopwords_lemma_L2 = SVMclassifier(train['transcription_After_RemovingStopWords_lemma'], train['label'], test['transcription_After_RemovingStopWords_lemma'], test['label'], 'l2')
SVM_nostopwords_lemma_L2.to_csv('./performance/coach_SVM_nostopwords_lemma_L2.csv', index=False)

# Logistic regression
def LRclassifier(train_text, train_label, test_text, test_label, regularizer):
    clf = Pipeline(
        [
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(penalty=regularizer))
        ]
    )
    clf.fit(train_text, train_label)
    prediction = clf.predict(test_text)
    matrix_report = metrics.classification_report(test_label, prediction, output_dict=True)
    matrix_report = pd.DataFrame(matrix_report).transpose()
    return matrix_report

LR_transcription = LRclassifier(train['transcription'], train['label'], test['transcription'], test['label'], 'none')
LR_transcription.to_csv('./performance/coach_LR_transcription.csv', index=False)

LR_transcription_L2 = LRclassifier(train['transcription'], train['label'], test['transcription'], test['label'], 'l2')
LR_transcription_L2.to_csv('./performance/coach_LR_transcription_L2.csv', index=False)

LR_nostopwords = LRclassifier(train['transcription_NoStopWords'], train['label'], test['transcription_NoStopWords'], test['label'], 'none')
LR_nostopwords.to_csv('./performance/coach_LR_nonstopwrods.csv', index=False)

LR_nostopwords_L2 = LRclassifier(train['transcription_NoStopWords'], train['label'], test['transcription_NoStopWords'], test['label'], 'l2')
LR_nostopwords_L2.to_csv('./performance/coach_LR_nonstopwrods_L2.csv', index=False)

LR_lemma = LRclassifier(train['transcription_afterLemmatization'], train['label'], test['transcription_afterLemmatization'], test['label'], 'none')
LR_lemma.to_csv('./performance/coach_LR_lemma.csv', index=False)

LR_lemma_L2 = LRclassifier(train['transcription_afterLemmatization'], train['label'], test['transcription_afterLemmatization'], test['label'], 'l2')
LR_lemma_L2.to_csv('./performance/coach_LR_lemma_L2.csv', index=False)

LR_nostopwords_lemma = LRclassifier(train['transcription_After_RemovingStopWords_lemma'], train['label'], test['transcription_After_RemovingStopWords_lemma'], test['label'], 'none')
LR_nostopwords_lemma.to_csv('./performance/coach_LR_nostopwords_lemma.csv', index=False)

LR_nostopwords_lemma_L2 = LRclassifier(train['transcription_After_RemovingStopWords_lemma'], train['label'], test['transcription_After_RemovingStopWords_lemma'], test['label'], 'l2')
LR_nostopwords_lemma_L2.to_csv('./performance/coach_LR_nostopwords_lemma_L2.csv', index=False)
