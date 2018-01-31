#Predict The Criminal

#https://www.hackerearth.com/challenge/competitive/predict-the-criminal/leaderboard/page/9/

import pandas as pd

#x = pd.read_csv('criminal_train.csv',
#usecols=['IIPRVHLT','IIOTHHLT','IFATHER'],delimiter=',')

x = pd.read_csv('criminal_train.csv',delimiter=',')
x = x.drop(['Criminal'], axis=1)
y = pd.read_csv('criminal_train.csv',usecols=['Criminal'],delimiter=',')


#Train-Test Split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=42)
y_train = y_train.values.ravel()


'''
#Deploying Model on Real Test set
x_train = x
y_train = y
y_train = y_train.values.ravel()
x_test = pd.read_csv('criminal_test.csv',delimiter=',')
'''

from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(x_train)
X_train_std = std_scale.transform(x_train)
X_test_std = std_scale.transform(x_test)


from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
model = DecisionTreeClassifier(max_depth=5)
#model = SVC(gamma=2, C=1)
model.fit(X_train_std,y_train)
y_pred = model.predict(X_test_std)


#Precision Score Evaluating on Test Data
import sklearn.metrics
#print sklearn.metrics.accuracy_score(y_test,y_pred)
precision_score = sklearn.metrics.precision_score(y_test,y_pred, average='micro')
print precision_score
#0.9343832021


'''
#Writing the Y_Predicted Value for the test set
output = pd.read_csv('criminal_test.csv',usecols=['PERID'])
output['Criminal'] = y_pred
output.to_csv('output.csv')
'''



