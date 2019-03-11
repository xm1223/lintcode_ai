# Movie Review Recognition
#
# https://www.lintcode.com/ai/movie-review-recognition
#
# TODO: not workable yet

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

train = pd.read_csv('~/ai/Movie/labeledTrainData.tsv', sep='\t')
unlabeled_train = pd.read_csv('~/ai/movie/unlabeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('~/ai/movie/testData.tsv', sep='\t')
#train.head()
#test.head()


def review_to_text(review, remove_stopwords):
    raw_text = BeautifulSoup(review, 'html').get_text()
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]

    return words


X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review, True)))

y_train = train['sentiment']

X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))

pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
params_count = {'count_vec__binary':[True, False], 'count_vec__ngram_range':[(1, 1), (1, 2)], 'mnb__alpha':[0.1, 1.0, 10.0]}
params_tfidf = {'tfidf_vec__binary':[True, False], 'tfidf_vec__ngram_range':[(1, 1), (1, 2)], 'mnb__alpha':[0.1, 1.0, 10.0]}
gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)
gs_count.fit(X_train, y_train)
#print(gs_count.best_score_)
#print(gs_count.best_params_)
count_y_predict = gs_count.predict(X_test)
gs_tfidf.fit(X_train, y_train)
#print(gs_tfidf.best_score_)
#print(gs_tfidf.best_params_)
tfidf_y_predict = gs_tfidf.predict(X_test)

submission_count = pd.DataFrame({'id': test['id'], 'sentiment': count_y_predict})
submission_tfidf= pd.DataFrame({'id': test['id'], 'sentiment': tfidf_y_predict})
submission_count.to_csv('~/av/movie/submission_count.csv', index=False)
submission_tfidf.to_csv('~/av/movie/submission_tfidf.csv', index=False)


import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_text(raw_sentence, False))

    return sentences


corpora = []
for review in unlabeled_train['review']:
    corpora += review_to_sentences(review.decode('utf8'), tokenizer)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 20   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(corpora, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = '~/ai/movie/300features_20minwords_10context"
model.save(model_name)

from gensim.models import Word2Vec
model = Word2Vec.load('~/ai/movie/300features_20minwords_10context')
model.most_similar('man')

import numpy as np


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype='float32')
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1

    return reviewFeatureVecs

clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append( review_to_text( review, remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append( review_to_text( review, remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

gbc = GradientBoostingClassifier()
params_gbc = {'n_estimators':[10, 100, 500], 'learning_rate':[0.01, 0.1, 1.0], 'max_depth': [2, 3, 4]}
gs = GridSearchCV(gbc, params_gbc, cv=4, n_jobs=-1, verbose=1)
gs.fit(trainDataVecs, y_train)
#print(gs.best_score_)
#print(gs.best_params_)
result = gs.predict(testDataVecs)
# Write the test results 
output = pd.DataFrame(data={'id':test['id'], 'sentiment':result} )
output.to_csv('~/ai/movie/submission_w2v.csv', index=False, quoting=3)
