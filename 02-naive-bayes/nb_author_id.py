import sys
from time import time
sys.path.append("../utils/")

from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.naive_bayes import GaussianNB

### create classifier
clf = GaussianNB()

### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)
print("train time:", round(time()-t0, 3), "s")

### use the trained classifier to predict labels for the test features
t0 = time()
pred = clf.predict(features_test)
print("test  time:", round(time()-t0, 3), "s")

### calculate and return the accuracy on the test data
from sklearn.metrics import accuracy_score
accuracy = clf.score(features_test, labels_test)
print(accuracy)