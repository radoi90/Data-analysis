import numpy as np
import pylab as P
import csv
#      0,      1,   2,        3,             4,        5,             6,  7
#Gesture,Control,Name,Score_knn,Score_knn_plus,Score_svm,Score_svm_plus,Avg
data = []
bins = range(-1,11)

#read data from csv file
with open('scores.csv', 'rb') as csvfile:
  datareader = csv.reader(csvfile, delimiter=',')
  for row in datareader:
    data.append(map(float, row))

data = np.array(data)
data = np.array([d for d in data if d[2] > 0])

P.figure(1)
P.title('KNN_all_on_correct')
P.hist(data[:,3],bins=bins)

P.figure(2)
P.title('KNN_all_on_all')
P.hist(data[:,4],bins=bins)

P.figure(3)
P.title('SVM_all_on_correct')
P.hist(data[:,5],bins=bins)

P.figure(4)
P.title('SVM_all_on_all')
P.hist(data[:,6],bins=bins)

P.figure(5)
P.title('AVG_all')
P.hist(data[:,6],bins=bins)

fig = 6
for g in range(1,11):
  gesture_data = np.array([d for d in data if d[0] == g])
  g = str(g)

  P.figure(fig)
  fig +=1
  P.title('KNN_'+g+'_on_correct')
  P.hist(gesture_data[:,3],bins=bins)

  P.figure(fig)
  fig +=1
  P.title('KNN_'+g+'_on_all')
  P.hist(gesture_data[:,4],bins=bins)

  P.figure(fig)
  fig +=1
  P.title('SVM_'+g+'_on_correct')
  P.hist(gesture_data[:,5],bins=bins)

  P.figure(fig)
  fig +=1
  P.title('SVM_'+g+'_on_all')
  P.hist(gesture_data[:,6],bins=bins)

  P.figure(fig)
  fig +=1
  P.title('AVG_'+g)
  P.hist(gesture_data[:,6],bins=bins)

P.show()
