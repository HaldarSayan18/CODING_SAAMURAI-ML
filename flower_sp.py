#PREDICTION OF FLOWER SPECIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# loading the dataset
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class_labels']
df = pd.read_csv('/content/iris.data', names = columns)
df.head(150)

#visualization of dataset
df.describe()
sns.pairplot(df, hue ='class_labels')

#separate i/o columns
data = df.values
x = data[:, 0:4]
y = data[:, 4]
print(y)

#split the data into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print(y_test)

#SVC(support vector machine) algorithm
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(x_train, y_train)

#calculation of the accuracy-1
predict1 = model_svc.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predict1) * 100)
for i in range(len(predict1)):
  print(y_test[i], predict1[i])

#logistic regression
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(x_train, y_train)

#calculation of the accuracy-2
predict2 = model_LR.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predict2) * 100)
for i in range(len(predict1)):
  print(y_test[i], predict1[i])

#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(x_train, y_train)

#calculation of the accuracy-3
predict3 = model_svc.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predict3))

#detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predict2))
