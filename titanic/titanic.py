# Titanic
#
# https://www.lintcode.com/ai/titanic
#

import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

train = pd.read_csv('~/ai/titanic/train.csv')
test = pd.read_csv('~/ai/titanic/test.csv')
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']
#print(X_train['Embarked'].value_counts())

# Preprocess the missing data
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
#print(X_train.info())

dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.fit_transform(X_test.to_dict(orient='record'))
#dict_vec.feature_names_


#model = RandomForestClassifier() # accuracy: 0.76
model = XGBClassifier() # accuracy: 0.77
cross_val_score(model, X_train, y_train, cv=5).mean()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('~/ai/titanic/submission.csv', index=False)
