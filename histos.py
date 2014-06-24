import numpy as np
import pylab as P
import csv

#      0,      1,   2,        3,        4,  5
#Gesture,Control,Name,Score_knn,Score_svm,Avg
data = []
bins = range(-1,12)
#read data from csv file
with open('scores.csv', 'rb') as csvfile:
  datareader = csv.reader(csvfile, delimiter=',')
  for row in datareader:
    data.append(map(float, row))

data = np.array(data)
controls = np.array([d for d in data if d[1] == 1 and d[0] != 1 and d[0] != 7])
p = np.array([d for d in data if d[1] == 0 and d[0] != 1 and d[0] != 7])
patients = np.copy(p)

for i in range (1,14):
    patients = np.vstack((patients, p))

P.figure(1)
P.subplot()
P.title('KNN')
P.hist([controls[:,3], patients[:,3]],bins=bins,align='right')

P.figure(3)
P.title('SVM')
P.hist([controls[:,4], patients[:,4]],range=[0,10],align='right')

P.figure(5)
P.title('AVG_all')
P.hist([controls[:,5], patients[:,5]],range=[0,10],align='right')

fig = 6
for g in range(1,11):
  f, (ax1, ax2, ax3) = P.subplots(1, 3, sharey=True)
  P.title(str(g))

  gesture_controls = np.array([d for d in controls if d[0] == g])
  gesture_patients = np.array([d for d in patients if d[0] == g])

  if gesture_patients.size > 0:
    ax1.set_title('KNN')
    ax1.hist([gesture_controls[:,3], gesture_patients[:,3]],bins=bins,align='right')

    ax2.set_title('SVM')
    ax2.hist([gesture_controls[:,4], gesture_patients[:,4]],bins=bins,align='right')

    ax3.set_title('AVG')
    ax3.hist([gesture_controls[:,5], gesture_patients[:,5]],bins=bins,align='right')

P.show()
