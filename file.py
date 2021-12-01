from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

faces = fetch_olivetti_faces(shuffle=True)
features = faces.data
labels = faces.target

train_features = features[:300]
test_features = features[300:]

train_labels = labels[:300]
test_labels = labels[300:]

#DISPLAY IMAGE
plt.imshow(train_features[0].reshape(64,64),cmap=plt.cm.gray, interpolation='nearest')
plt.show()

#we will check if the given person is person with lable 13
train_labels_acc = (train_labels==13)
test_labels_acc = (test_labels==13)

#TRAINING MODEL USING LOGISTIC REGRESSION
clf = LogisticRegression()
clf.fit(train_features,train_labels_acc)
predicted = clf.predict(test_features)

#CROSS VALIDATION
a = cross_val_score(clf, train_features, train_labels_acc, cv=3, scoring="accuracy")
print(a.mean())

#Printing Mean Squared error
print("mean squared error: ",mean_squared_error(test_labels,predicted))

