#===============importing needed tools=========================#
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
"""import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize"""
import string
#===============importing needed tools=========================#


# Download the stopwords corpus from NLTK
"""nltk.download('stopwords')
nltk.download('punkt')"""

#=======================load data & remove unnamed columns ==============================#
messages = pd.read_csv("spam.csv", encoding='latin-1')
messages = messages.loc[:, ~messages.columns.str.contains('^Unnamed')]
#=======================load data & remove unnamed columns ==============================#


messages['type'] = messages['type'].map({
    'ham' : 0,
    'spam' : 1
    })

# Define the preprocessing function
def preprocess_message(message):
    
    Message = message.lower()
    Message = re.sub('\[.*?\]', '', Message)
    Message = re.sub('https?://\S+|www\.\S+', '', Message)
    Message = re.sub('<.*?>+', '', message)
    Message = re.sub('[%s]' % re.escape(string.punctuation), '', Message)
    Message = re.sub('\n', '', Message)
    Message = re.sub('\w*\d\w*', '', Message)
    
    return message


messages['message'] = messages['message'].apply(preprocess_message)


X = messages['message']
Y = messages['type']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

params = {
    'criterion':  ['gini', 'entropy'],
    'max_depth':  [None, 2, 4, 6, 8, 10],
    'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    'splitter': ['best', 'random']
}

model = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1,
)


model.fit(xv_train, y_train)

y_pred_train = model.predict(xv_train)
y_pred = model.predict(xv_test)


print("Accuracy train:", accuracy_score(y_train, y_pred_train))
print("Accuracy test:", accuracy_score(y_test, y_pred))


#print("confusion matrix:", confusion_matrix(y_test, y_pred))


""""#=======================load data & remove unnamed columns ==============================#
new_msg = pd.read_csv("new_data.csv", encoding='latin-1')
new_msg = new_msg.loc[:, ~new_msg.columns.str.contains('^Unnamed')]
#=======================load data & remove unnamed columns ==============================#

new_msg['message'] = new_msg['message'].apply(preprocess_message)


new_msg['type'] = new_msg['type'].map({
    'ham' : 0,
    'spam' : 1
    }) 

x_new = new_msg['message']
y_new = new_msg['type']

xv_new = vectorization.transform(x_new)

y_pred_new = model.predict(xv_new)



print("test new:")
print(y_pred_new)
print("Accuracy new:", accuracy_score(y_new, y_pred_new))"""
