# Spam Message Classification
#
# https://www.lintcode.com/ai/spam-message-classification
#
# other solutions: https://github.com/yuweiming70/lintcode-AI/blob/master/%E5%9E%83%E5%9C%BE%E7%9F%AD%E4%BF%A1%E5%88%86%E7%B1%BB.py
#

import numpy as np
import pandas as pd
import csv
import os
import re
import nltk.stem as stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

stemmer = stem.snowball.EnglishStemmer()

def clean_text(texts):
    text_list = []
    for text in texts:
        # 将单词转换为小写
        text = text.lower()
        # 删除非字母、数字字符
        text = re.sub(r'[^a-z\']', ' ', text)
        # 恢复常见的简写
        text = re.sub(r'what\'s', 'what is ', text)
        text = re.sub(r'\'s', ' ', text)
        text = re.sub(r'\'ve', ' have ', text)
        text = re.sub(r'can\'t', 'can not ', text)
        text = re.sub(r'cannot', 'can not ', text)
        text = re.sub(r'n\'t', ' not ', text)
        text = re.sub(r'\'m', ' am ', text)
        text = re.sub(r'\'re', ' are ', text)
        text = re.sub(r'\'d', ' will ', text)
        text = re.sub(r'ain\'t', ' are not ', text)
        text = re.sub(r'aren\'t', ' are not ', text)
        text = re.sub(r'couldn\'t', ' can not ', text)
        text = re.sub(r'didn\'t', ' do not ', text)
        text = re.sub(r'doesn\'t', ' do not ', text)
        text = re.sub(r'don\'t', ' do not ', text)
        text = re.sub(r'hadn\'t', ' have not ', text)
        text = re.sub(r'hasn\'t', ' have not ', text)
        text = re.sub(r'\'ll', ' will ', text)
        #进行词干提取
        new_text = ''
        for word in word_tokenize(text):
            new_text = new_text + ' ' + stemmer.stem(word)

        text_list.append(new_text)
    return text_list

def read_cvs(file):
    data_list = list(csv.reader(open(os.path.expanduser(file), encoding='utf-8')))
    lines = len(data_list)
    label = np.zeros([lines - 1, ])
    content = []
    i = 0
    for data in data_list:
        if data[0] == 'Label' or data[0] == 'SmsId':
            continue
        if data[0] == 'ham':
            label[i] = 0
        if data[0] == 'spam':
            label[i] = 1
        content.append(data[1])
        i += 1

    #print(train_data_label.shape, len(train_data_content))
    return label, content

# load data
y_train, train_data_content = read_cvs('~/ai/Spam/train.csv')
_, test_data_content = read_cvs('~/ai/Spam/test.csv')
train_data_content = clean_text(train_data_content)
test_data_content = clean_text(test_data_content)

# TF-IDF
all_text_list = list(train_data_content) + list(test_data_content)
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',token_pattern=r'\w{1,}',
                              max_features=5000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(all_text_list)
X_train = text_vector.transform(train_data_content).toarray()
X_test = text_vector.transform(test_data_content).toarray()
#print(X_train.shape, X_test.shape, type(X_train))

# train model
model = LogisticRegression(C=100.0)
model.fit(X_train, y_train)
train_scores = model.score(X_train, y_train)
#print(train_scores)
predictions = model.predict_proba(X_test)
#print(predictions.shape)

submission = pd.read_csv('~/ai/Spam/sampleSubmission.csv')
for i in range(predictions.shape[0]):
    if predictions[i, 0] < 0.5:
        submission.loc[i, 'Label'] = 'spam'
    else:
        submission.loc[i, 'Label'] = 'ham'

submission.to_csv('~/ai/Spam/submission.csv', index=False)  # not store index
