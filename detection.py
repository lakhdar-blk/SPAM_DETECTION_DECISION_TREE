#===============importing needed tools=========================#
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import string
#===============importing needed tools=========================#


# Download the stopwords corpus from NLTK
nltk.download('stopwords')
nltk.download('punkt')

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
    
    stop_words = set(stopwords.words('english'))
    message = ' '.join([word for word in message.split() if word not in stop_words])
    return message


messages['message'] = messages['message'].apply(preprocess_message)


X = messages['message']
Y = messages['type']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


vectorizer = CountVectorizer()
spam_features = vectorizer.fit_transform(messages['message'])

params = {
    'criterion':  ['gini', 'entropy'],
    'max_depth':  [None, 2, 4, 6, 8, 10],
    #'max_depth':  [None, 10, 20, 30, 40, 50,60 , 70, 80, 90],
    'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    'splitter': ['best', 'random']
}


X_train, X_test, y_train, y_test = train_test_split(spam_features, messages['type'], test_size=0.2)

# Create a decision tree classifier with some improvements
model = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1,
)

# Train the classifier on the training data
model.fit(X_train, y_train)

# Evaluate the classifier on the testing data
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


y_pred = model.predict(X_test)
#print("Accuracy test:", accuracy_score(y_test, y_pred))
print("confusion matrix:", confusion_matrix(y_test, y_pred))


#================================test with unseen data==================================================#

#=======================load data & remove unnamed columns ==============================#
new_msg = pd.read_csv("new_data.csv", encoding='latin-1')
new_msg = new_msg.loc[:, ~new_msg.columns.str.contains('^Unnamed')]
#=======================load data & remove unnamed columns ==============================#

new_msg['message'] = new_msg['message'].apply(preprocess_message)

spam_features_new = vectorizer.transform(new_msg['message'])

y_pred_new = model.predict(spam_features_new)

print(y_pred_new)

#================================test with unseen data==================================================#