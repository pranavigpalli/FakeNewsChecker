import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# I am trying to teach myself AI basics to prepare a little for specializing in UCI's 
# Intelligent Systems Specialization. Below is just some of my first work in understanding basic concepts.

# read CSV data onto a dataframe
data = pd.read_csv('fake_or_real_news.csv')

# lambda function returns 0 if the article is real and 1 if it's fake
data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)

# removing the label column (hence the axis = 1) 
data = data.drop('label', axis = 1)

# assigns the text column to x (for input data) and the fake column to y (output data)
x, y = data['text'], data['fake']

# trains represent the training data, while tests represent the testing data
# 20% of the data will be used for testing, while the remaining 80% will be used for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# initializes a TfidfVectorizer object, removes common English words
# ignores terms that appear in more than 70% of the documents
vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

# fits the TfidfVectorizer to the training data (x_train) and transforms text data into a TF-IDF matrix 
# (matrix where rows are documents and columns are terms)
x_train_vectorized = vectorizer.fit_transform(x_train)

# transforms the test data (x_test) using the already-fitted TfidfVectorizer
x_test_vectorized = vectorizer.transform(x_test)

# initializes a LinearSVC model, which tries to find the best hyperplane that separates the 
# two classes (REAL vs FAKE) in the feature space
clf = LinearSVC()

# trains the LinearSVC model on the vectorized training data (x_train_vectorized) and 
# the corresponding labels (y_train)
clf.fit(x_train_vectorized, y_train)

# evaluates the model's performance on the test data by calculating the accuracy score
print(clf.score(x_test_vectorized, y_test))