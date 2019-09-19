import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

SPECIES_MAPPING = {'Iris-setosa': 0,
                   'Iris-versicolor': 1,
                   'Iris-virginica': 2}

def create_labels(val):
    return SPECIES_MAPPING[val]

colnames=['petal_length', 'petal_width', 'sepal_length','sepal_width','species']
data1 = pd.read_csv("iris.csv", names=colnames, header=None)
data1 = shuffle(data1, random_state=11)
data1['species'] = data1['species'].apply(create_labels)

x = []
y = []
for row_data in data1.iterrows():
    row = row_data[1]
    x.append([row['petal_length'], row['petal_width'], row['sepal_length'], row['sepal_width']])
    y.append(row['species'])
x = np.array(x)
y = np.array(y)
train_data = x[:119]
test_data = x[119:]
train_labels = y[:119]
test_labels = y[119:]
#print(len(train_data), len(test_data), len(train_labels), len(test_labels))


model = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(4,)),
 tf.keras.layers.Dense(50, activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(50, activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
class myCallback(tf.keras.callbacks.Callback):

    def __init__(self, model):
        self.model = model
        self.collectedData = []

    def on_epoch_end(self, epoch, logs=None):
        if(epoch % MAX_EPOCHS == 0):
            eval = self.model.evaluate(test_data, test_labels, verbose=0)
            print('Loss for epoch {:4d} is {:7.3f} and accuracy is {:7.3f} Eval loss={:7.3f} Eval acc={:7.3f}'.format(epoch, logs['loss'], logs['acc'], eval[0], eval[1]))
            self.collectedData.append([epoch, logs['loss'], logs['acc'], eval[0], eval[1]])


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

MAX_EPOCHS = 60
callback = myCallback(model)
train_result = model.fit(train_data, train_labels, epochs=MAX_EPOCHS, batch_size=20, verbose=9, callbacks=[callback], validation_data=(test_data, test_labels))
eval_result = model.evaluate(test_data, test_labels)

petal_lengh = x[:, 0]
petal_width = x[:, 1]
sepal_lenth = x[:, 2]
sepal_width = x[:, 3]



subPlotsShape = 330  # 3x3 figure
fig = plt.figure(figsize=(12, 9))
colors = ['r', 'g', 'b']
plt.subplot(3, 3, 1)
plt.scatter(petal_lengh, petal_width)
plt.subplot(3,3,2)
plt.scatter(petal_lengh, sepal_width)
plt.subplot(3,3,3)
plt.scatter(sepal_lenth, petal_width)
plt.subplot(3,3,4)
plt.scatter(petal_lengh, sepal_lenth)
plt.subplot(3,3,5)
plt.scatter(sepal_lenth, sepal_width)
plt.subplot(3,3,6)
plt.scatter(petal_width, sepal_width)
plt.subplot(3,3,7)
plt.plot(train_result.history['val_loss'])
plt.subplot(3,3,8)
plt.plot(train_result.history['loss'])
plt.plot(train_result.history['val_loss'])
plt.subplot(3,3,9)
plt.plot(train_result.history['acc'])
plt.plot(train_result.history['val_acc'])


#plt.scatter(petal_lengh, petal_width)
plt.show()

#plt.plot(model.loss, MAX_EPOCHS)
'''
print(train_result.history)
loss = train_result.history['loss']
acc = train_result.history['acc']
for idx in range(len(loss)):
    print("Epoch {} : loss={}, acc={}".format(idx, loss[idx], acc[idx]))
'''


