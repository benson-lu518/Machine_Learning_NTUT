from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
#print(iris.feature_names)
#print(iris.target_names)

data_x=iris.data
data_y=iris.target

x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test) 

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))
