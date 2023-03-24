import tensorflow as tf
from tensorflow import keras
from keras.metrics import Accuracy, AUC, AU
from keras import layers
import numpy as np
from tensorflow.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import datetime
from model.data import Dataset
from sklearn.model_selection import train_test_split

ds = Dataset(['cnv', 'gene_expression', 'snv', 'methylation'])
response = ds.response
data = ds.return_feature()
labels = response['LN_IC50'].values
scaler = MinMaxScaler()
labels = labels.reshape(-1,1)
scaler.fit(labels)
labels = scaler.transform(labels)

y = []
for i in labels:
    if (i[0]<=0.5):
        y.append(1)
    else:
        y.append(0)
labels = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, 
                                                    random_state=40)

# 435+810 = 1245
input_shape = (1245,)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(units=1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(units=128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(.5),
        layers.Dense(1, activation='sigmoid')
    ]
)
batch_size = 32
epochs = 100
# 64 100 0.85

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=
              [
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                AUC(),
                Accuracy(),
              ]
            )


model.fit(
    x=X_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=epochs,
    validation_split=.1,
    callbacks=[tensorboard_callback]
    )

    

        