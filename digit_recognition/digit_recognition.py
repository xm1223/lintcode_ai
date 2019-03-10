# Digit Recognition
#
# https://www.lintcode.com/ai/digit-recognition
#
# other solutions: https://blog.csdn.net/searcher_recommeder/article/details/79482137
# 

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import SGDClassifier

# Load data
train = pd.read_csv('~/ai/DigitRecognition/train.csv')
test = pd.read_csv('~/ai/DigitRecognition/test.csv')
# train.shape
# test.shape
# print(train.sample(1))

# Drop the label column as train set
X_train = train.drop('label', 1)
# Separate lable column as train target
y_train = train['label']
X_test = test

# Change to different model or hyperparameters
#model = SGDClassifier() # accuracy 0.88
model = RandomForestClassifier() # accuracy 0.95
# Train model
model.fit(X_train, y_train)
# Predict via model
predictions = model.predict(X_test)

submission = pd.DataFrame({'ImageId': range(1, predictions.shape[0] + 1),'Label': predictions})
submission.to_csv('~/ai/DigitRecognition/submission.csv', index=False)
