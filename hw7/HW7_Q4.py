import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
data_x=iris.data
data_y=iris.target



data_y= pd.get_dummies(data_y).values

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



#avg accuracy for 10 times
loss_sum=0
accuracy_sum=0
for i in range(10):


    model.fit(x_train, y_train, batch_size=50, epochs=100)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    loss_sum+=loss
    accuracy_sum+=accuracy
   

loss=loss_sum/10.0
accuracy=accuracy_sum/10.0
print ("Accuracy : ", accuracy)
print ("Loss : ", loss)



