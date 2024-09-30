from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

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

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(x_train, y_train)



#avg accuracy for 10 times
count=0
for i in range(10):
    y_pred = classifier.predict(x_test) 
    count+=accuracy_score(y_test, y_pred)
    
count=count/10
print ("Accuracy : ", count)




