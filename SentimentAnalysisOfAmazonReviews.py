# -*- coding: utf-8 -*-
"""
AIM : Design, save and load model for sentiment analysis using dataset from amazon shopping site
"""

import numpy as np

#importing library to read data
import pandas as pd
data=pd.read_csv("/content/amazon_cells_labelled.txt",header=0,sep='\t')
data

#splitting data to x and y (input and output)
x=data.iloc[:,0]
y=data.iloc[:,-1]

#importing NLP library
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english')

#transforming input data and fit to count vectorizer
cv.fit(x)
x_cv=np.array(cv.transform(x).todense())
print(x_cv)

#converting to dense and array
# print("---------To Dense----------")
# print(x_cv.todense())
# dense_cv=x_cv.todense()
# print("---------To Array-----------")
# print(x_cv.toarray())

#tranforming count vector to tfidf format
from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer()
tfidf.fit(x_cv)
x_tfidf=np.array(tfidf.transform(x_cv).todense())
print(x_tfidf)

# dense of tfidf and array of tfidf
# print("---------to Dense-----------")
# dense_tfidf=x_tfidf.todense()
# print(dense_tfidf)
# print("---------to Array------------")
# print(x_tfidf.toarray())

#splitting training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(x_tfidf,y ,test_size=0.25, shuffle=True)

#using SVM as two class classification model
from sklearn.svm import SVC
classify = SVC(kernel='linear')

#fitting data to our model
classify.fit(X_train , y_train)

#predicting output with testing data
y_pred=classify.predict(X_test)
print(y_pred)

#creating a confusion matrix with data
from sklearn import metrics
import matplotlib.pyplot as plt
cm = metrics.confusion_matrix(y_test,y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
cm_display.plot()
plt.show()

cm

#calculating accuracy and precision of model with confusion matrix
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))
print("Precision : ",metrics.precision_score(y_test,y_pred))

#dump model
import pickle
pickle.dump(classify,open("demo_svm","wb"))
pickle.dump(cv,open("demo_cv","wb"))
pickle.dump(tfidf,open("demo_tfidf","wb"))

#load dumped models
classify = pickle.load(open("demo_svm","rb"))
countvec = pickle.load(open("demo_cv","rb"))
TFIDF = pickle.load(open("demo_tfidf","rb"))

#use loaded modules for sentiment prediction of input.
sentiment_prediction = ['product was bad','excellent product','wow! amazing product']
sentiment_prediction = pd.Series(sentiment_prediction)
sentiment_prediction = np.array((countvec.transform(sentiment_prediction)).todense())
sentiment_prediction = np.array((TFIDF.transform(sentiment_prediction)).todense())
output = classify.predict(sentiment_prediction)

print(output) #Final output of sentiment prediction model testing
