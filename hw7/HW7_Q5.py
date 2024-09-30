from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LSTM,GRU
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
# 寫一個LossHistory類，保存loss和acc
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("mnist_keras.png")
        plt.show()

# 訓練參數
learning_rate = 0.001
epochs = 20
batch_size = 512
n_classes = 10

# 定義圖像維度reshape
img_rows, img_cols = 28, 28


# 加載keras中的mnist數據集 分爲60,000個訓練集，10,000個測試集
# 將100張RGB，3通道的16*32彩色圖表示爲(100,16,32,3)，（樣本數，高，寬，顏色通道數）
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# RNN shape
# x_train = x_train.reshape(-1, img_rows, img_cols)
# x_test = x_test.reshape(-1, img_rows, img_cols)
# CNN shape
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# 將X_train, X_test的數據格式轉爲float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 將X_train, X_test歸一化0-1
x_train /= 255
x_test /= 255

# 輸出0-9轉換爲ont-hot形式
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

# 建立模型
model = Sequential()


# lenet-5
model.add(Convolution2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(img_rows, img_cols, 1), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))


model.add(Dense(n_classes, activation='softmax'))


#打印模型# verbose=1顯示進度條
model.summary()
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',metrics=['accuracy'])
history = LossHistory()
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs, verbose=1,validation_data=(x_test, y_test),callbacks=[history])
model.save('rnn_weight.h5')


# 測試
# model.load_weights('rnn_weight.h5')

y_predict = model.predict(x_test, batch_size=512, verbose=1)
# y_predict = (y_predict > 0.007).astype(int)
y_predict = (y_predict > 0.01).astype(int)
y_true = np.reshape(y_test, [-1])
y_pred = np.reshape(y_predict, [-1])

# 評價指標
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average='binary')
f1score = f1_score(y_true, y_pred, average='binary')

# Micro F1: 將n分類的評價拆成n個二分類的評價，將n個二分類評價的TP、FP、RN對應相加，計算評價準確率和召回率，由這2個準確率和召回率計算的F1 score即爲Micro F1。
# Macro F1: 將n分類的評價拆成n個二分類的評價，計算每個二分類的F1 score，n個F1 score的平均值即爲Macro F1。
# 一般來講，Macro F1、Micro F1高的分類效果好。Macro F1受樣本數量少的類別影響大。
micro_f1 = f1_score(y_true, y_pred,average='micro')
macro_f1 = f1_score(y_true, y_pred,average='macro')


print('accuracy:',accuracy)
print('precision:',precision)
print('recall:',recall)
print('f1score:',f1score)
print('Macro-F1: {}'.format(macro_f1))
print('Micro-F1: {}'.format(micro_f1))

#繪製訓練的acc-loss曲線
history.loss_plot('epoch')
