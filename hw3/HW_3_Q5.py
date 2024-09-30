from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import accuracy_score 
data = load_breast_cancer()

data_x=data.data
data_y=data.target

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

#x_train x_text x_val
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=1-train_ratio)
x_train,x_val,y_train,y_val=train_test_split(data_x,data_y,test_size=test_ratio/(test_ratio+validation_ratio))

#feature selection
select=SelectKBest(f_classif,k=5)
select.fit(x_val,y_val)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_val, y_val)
dataframe=select.get_support(True)

x_train=x_train[:,[dataframe[0],dataframe[1],dataframe[3],dataframe[4]]]

#avg accuracy for 10 times
count=0
for i in range(10):
    y_pred = classifier.predict(x_test) 
    count+=accuracy_score(y_test, y_pred)
    
count=count/10
print ("Accuracy : ", count)

