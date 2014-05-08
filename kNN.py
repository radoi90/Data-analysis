import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier

data = []

#read data from csv file
with open('correct_gestures_plus.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	for row in datareader:
		data.append(map(float, row))

data = np.array(data)

#scale data s.t all features have zero mean and unit variance
gestures_X = data[:,1:]
gestures_y = np.array(map(int,data[:,0]))
#gestures_X_scaled = preprocessing.scale(gestures_X)

#split the data in training and test sets
gestures_X_train, gestures_X_test, gestures_y_train, gestures_y_test = \
	cross_validation.train_test_split(gestures_X, gestures_y, test_size=0.2, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'n_neighbors': range(2,11)}]

scores = ['precision', 'recall']

for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()

		clf = GridSearchCV(KNeighborsClassifier(n_neighbors=1), tuned_parameters, cv=5, scoring=score)
		clf.fit(gestures_X_train, gestures_y_train)

		print("Best parameters set found on development set:")
		print()
		print(clf.best_estimator_)
		print()
		print("Grid scores on development set:")
		print()
		for params, mean_score, scores in clf.grid_scores_:
				print("%0.3f (+/-%0.03f) for %r"
							% (mean_score, scores.std() / 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = gestures_y_test, clf.predict(gestures_X_test)
		print(classification_report(y_true, y_pred))
		print()

clf = clf.best_estimator_
classifiers = OneVsRestClassifier(clf, n_jobs=-1)
classifiers.fit(gestures_X_train, gestures_y_train)

#score each gesture by using the probability of belonging to its class
scores = []
for i in range(0,len(gestures_y)):
	#pick the estimator for the correct class
	class_estimator = classifiers.estimators_[gestures_y[i]-1]
	#get the probability of belonging to that class (being labeled 1)
	prob = class_estimator.predict_proba(gestures_X_scaled[i])[0][1]
	scores.append(10*prob)

with open('s2knn_scores.csv', 'wb') as csvfile:
    scorewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for score in scores:
			scorewriter.writerow([int(score)])
