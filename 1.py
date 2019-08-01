
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
documents = pd.read_table('SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])

# Output printing out first 5 rows
documents.head()

# Change the lable coulumn's values to numbers, 0 for ham and 1 for spam to be exact
documents['label'] = documents.label.map({'ham':0, 'spam':1})
print(documents.shape)
documents.head() # returns (rows, columns)


# Creating an arbitrary document for exp purpose
#documents = ['Hello, how are you!', 'Win money, win from home.', 'Call me now.', 'Hello, Call hello you tomorrow?']


# Converting the data now to a matrix of features.
docArray = countVector.transform(documents['sms_message']).toarray()
docArray

# Converting the data to a dataframe of frequency.
freqMat = pd.DataFrame(docArray, columns = countVector.get_feature_names())

# Data Preprocessing and splitting
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(documents['sms_message'], documents['label'], random_state = 1)

print('Number of rows in the total set: {}'.format(documents.shape[0]))
print('Number of rows in the training set: {}'.format(xTrain.shape[0]))
print('Number of rows in the test set: {}'.format(xTest.shape[0]))

# Applying Bag of words to the dataset
# Instantiate the CountVectorizer method
countVector = CountVectorizer()

# Fit the training data and then return the matrix
traininData = countVector.fit_transform(xTrain)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testingData = countVector.transform(xTest)

from sklearn.naive_bayes import MultinomialNB
nBayes = MultinomialNB()
nBayes.fit(traininData, yTrain)

predictions = nBayes.predict(testingData)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(yTest, predictions)))
print('Precision score: ', format(precision_score(yTest, predictions)))
print('Recall score: ', format(recall_score(yTest, predictions)))
print('F1 score: ', format(f1_score(yTest, predictions)))