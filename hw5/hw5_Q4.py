
import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

total=0
for i in range(10):
        
    iris = datasets.load_iris()
    data_x = iris.data
    data_y= iris.target

    pca = decomposition.PCA(n_components=3)
    pca.fit(data_x)

    data_x= pca.transform(data_x)
    data_y= np.choose(data_y, [1, 2, 0]).astype(float)

    x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.3)

    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train,y_train)
    acc=knn.score(x_test,y_test)
    print('Acc=%.2f'%acc)
    total+=acc
avg=total/10
print('avg acc=%.2f'%avg)

