import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

labels = [23, 24, 23, 22, 22, 22, 22, 21, 23, 22]
data = []

with open('g10.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	for row in datareader:
		data.append(map(float, row))

data = np.array(data)

from sklearn import preprocessing

gestures_X = preprocessing.scale(data[:,1:])
gestures_y = []
gesture = 1

for i in range (0,10):
	for j in range(0,labels[i]):
		#gestures_y.append(i+1)
		gestures_y.append(1 if i == gesture else 0)
gestures_y = np.array(gestures_y)

gestures_X_train, gestures_X_test, gestures_y_train, gestures_y_test = cross_validation.train_test_split(gestures_X, gestures_y, test_size=0.1)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(gestures_X_train, gestures_y_train)
print knn.predict_proba(gestures_X_test)[:,1]
print knn.predict(gestures_X_test)
print knn.score(gestures_X_test, gestures_y_test)
print gestures_y_test
