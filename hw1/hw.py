from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

iris = datasets.load_iris()
data_x=iris.data
data_y=iris.target

k=0
highest=0
for j in range(3,12):
    total=0
    for i in range(10):

        x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=1-train_ratio)
        x_train,x_val,y_train,y_val=train_test_split(data_x,data_y,test_size=test_ratio/(test_ratio+validation_ratio))

        knn=KNeighborsClassifier(n_neighbors=j)
        knn.fit(x_train,y_train)
        acc=knn.score(x_val,y_val)
        total+=acc
    avg=total/10
    if (avg>highest):
        highest=avg
        k=j

total_test=0
for i in range(10):
    x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=1-train_ratio)
    x_train,x_val,y_train,y_val=train_test_split(data_x,data_y,test_size=test_ratio/(test_ratio+validation_ratio))


    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    acc=knn.score(x_test,y_test)
    total_test+=acc
avg_test=total_test/10
#print('acc:%.2f, k=%d'%(avg_test,k))