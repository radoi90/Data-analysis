import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
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
gestures_X_scaled = preprocessing.scale(gestures_X)

#split the data in training and test sets
gestures_X_train, gestures_X_test, gestures_y_train, gestures_y_test = \
  train_test_split(gestures_X_scaled, gestures_y, test_size=0.2, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'gamma': [1e-3, 1e-4],
                      'C': [1, 10, 100, 1000], 'degree': [2,3,4],
                      'coef0': [0, 1]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
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
clf.set_params(probability=True)

classifiers = OneVsRestClassifier(clf, n_jobs=-1)
classifiers.fit(gestures_X_train, gestures_y_train)

#score each gesture by using the probability of belonging to its class
splits=[0, 0.1, 0.25, 0.35, 0.5, 0.8, 1.1, 1.75, 3.3, 5, 10]
scores = []
for i in range(0,len(gestures_y)):
  #pick the estimator for the correct class
  class_estimator = classifiers.estimators_[gestures_y[i]-1]
  #get the probability of belonging to that class (being labeled 1)
  log = 1/-class_estimator.predict_log_proba(gestures_X_scaled[i])[0][1]
  score = 0
  for j in range(0,len(splits)):
    if log >= splits[j]:
      score = j
  scores.append(score)

with open('s2svm_scores.csv', 'wb') as csvfile:
    scorewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for score in scores:
      scorewriter.writerow([score])
