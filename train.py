import tensorflow as tf
from tensorflow import keras
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras import Model
from keras.metrics import Accuracy, AUC
import pandas as pd
from keras.models import Sequential
from keras import layers
import numpy as np
from tensorflow.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = np.load('./data/processed_data/all_feature.npy')
response = pd.read_csv('./data/processed_data/response.csv', sep='\t')
#features = tf.constant(data)
labels = response['LN_IC50'].values
scaler = MinMaxScaler()
labels = labels.reshape(-1,1)
scaler.fit(labels)
labels = scaler.transform(labels)
# labels = np.squeeze(1)
#labels = tf.constant(labels)
# dataset = Dataset.from_tensors((features, labels))

y = []
for i in labels:
    if (i[0]<=0.6):
        y.append(1)
    else:
        y.append(0)
labels = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, 
                                                    random_state=40)

# DATASET_SIZE = int(dataset.__len__())

# train_size = int(0.7 * DATASET_SIZE)
# val_size = int(0.15 * DATASET_SIZE)
# test_size = int(0.15 * DATASET_SIZE)

# # dataset = dataset.shuffle()
# train_dataset = dataset.take(train_size)
# test_dataset = dataset.skip(train_size)
# val_dataset = test_dataset.skip(val_size)
# test_dataset = test_dataset.take(test_size)



input_shape = (1301,)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(units=1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(units=128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid'),
    ]
)

batch_size = 256
epochs = 100


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=[
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    AUC()])


model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs,validation_split=.1)

    

        