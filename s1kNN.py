import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier

data = []

#read data from csv file
with open('correct_gestures_plain.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	for row in datareader:
		data.append(map(float, row))

data = np.array(data)

#scale data s.t all features have zero mean and unit variance
gestures_X = data[:,1:]
gestures_y = np.array(map(int,data[:,0]))
gestures_X_scaled = preprocessing.scale(gestures_X)

#split the data in training and test sets
gestures_X_train, gestures_X_test, gestures_y_train, gestures_y_test = \
	cross_validation.train_test_split(gestures_X_scaled, gestures_y, test_size=0.1, random_state=0)

#use k-fold cross validation to determine the best N for KNN
best_mean = 0
best_std = 0
best_N = 0
print "Beginning KNN k-fold cross-validation"
for n in range(2,10):
	knn = KNeighborsClassifier(n_neighbors=n)
	#use one vs. rest strategy
	classifiers = OneVsRestClassifier(knn, n_jobs=-1)
	classifiers.fit(gestures_X_train, gestures_y_train)

	#get the cross validated score
	scr = cross_validation.cross_val_score(classifiers, gestures_X_train, gestures_y_train, cv = 5)
	if (scr.mean() > best_mean) or (scr.mean() == best_mean and scr.std() < best_std):
		best_mean = scr.mean()
		best_std = scr.std()
		best_N = n
	print("n=%d\nAccuracy: %0.3f (+/- %0.2f)" % (n,scr.mean(), scr.std()*2))

print ("Using KNN with N=%d" % (best_N))
#use KNN with n = 5, determined through n-fold cross validation
knn = KNeighborsClassifier(n_neighbors=best_N)
#use one vs. rest strategy
classifiers = OneVsRestClassifier(knn, n_jobs=-1)
classifiers.fit(gestures_X_train, gestures_y_train)

#score each gesture by using the probability of belonging to its class
scores = []
index = 0
for i in range(0,len(gestures_y)):
	#pick the estimator for the correct class
	class_estimator = classifiers.estimators_[gestures_y[i]-1]
	#get the probability of belonging to that class (being labeled 1)
	prob = class_estimator.predict_proba(gestures_X_scaled[i])[0][1]
	scores.append(10*prob)

with open('s1knn_scores.csv', 'wb') as csvfile:
    scorewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for score in scores:
			scorewriter.writerow([int(score)])
