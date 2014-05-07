import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import preprocessing

data = []

#read data from csv file
with open('g10.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	for row in datareader:
		data.append(map(float, row))

data = np.array(data)

#scale data s.t all features have zero mean and unit variance
gestures_X = data[:,1:]
gestures_X_scaled = preprocessing.scale(gestures_X)

#train a 1vsAll KNN classifier for each gesture
labels = [23, 24, 23, 22, 22, 22, 22, 21, 23, 22]
classifiers=[]
for n in range (2, 11):
	m = []
	s = []
	for gesture in range(0,10):
		#compute the labels
		gestures_y = []
		for i in range (0,10):
			for j in range(0,labels[i]):
				#gestures_y.append(i+1)
				gestures_y.append(1 if i == gesture else 0)
		gestures_y = np.array(gestures_y)

		gestures_X_train, gestures_X_test, gestures_y_train, gestures_y_test = cross_validation.train_test_split(gestures_X_scaled, gestures_y, test_size=0.1, random_state=0)

		knn = KNeighborsClassifier(n_neighbors=n)
		scr = cross_validation.cross_val_score(knn, gestures_X_train, gestures_y_train, cv = 5)
		m.append(scr.mean())
		s.append(scr.std()*2)
		classifiers.append(knn)
		knn.fit(gestures_X_train, gestures_y_train)
	m = np.array(m)
	s = np.array(s)
	print("n=%d\nAccuracy: %0.3f (+/- %0.2f)" % (n,m.mean(), s.sum()))

	#score each gesture by using the probability of belonging to its class
	scores = []
	index = 0
	for i in range(0,10):
		for j in range(0,labels[i]):
			scores.append(int(10*classifiers[i].predict_proba(gestures_X_scaled[index])[0][1]))
			index += 1

with open('s1knn_scores.csv', 'wb') as csvfile:
    scorewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for score in scores:
			scorewriter.writerow([score])
