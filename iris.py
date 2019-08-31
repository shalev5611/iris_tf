import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


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


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10, batch_size=5, verbose=9)
model.evaluate(test_data, test_labels)




