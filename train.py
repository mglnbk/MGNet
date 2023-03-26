import tensorflow as tf
from tensorflow import keras
from keras.metrics import Accuracy, AUC
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.metrics import Precision, Recall
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
labels = response['AUC'].values
# scaler = MinMaxScaler()
# labels = labels.reshape(-1,1)
# scaler.fit(labels)
# labels = scaler.transform(labels)

y = []
for i in labels:
    if (i<=0.6):
        y.append(1)
    else:
        y.append(0)
labels = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, 
                                                    random_state=40)

model = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[1],)),
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

def scheduler(epoch, lr):
    if(epoch % 50 ==0 and epoch !=0):
        return lr*0.1
    else:
        return lr
reduce_lr = LearningRateScheduler(scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience=10)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
              loss="binary_crossentropy",
              metrics=
              [
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(curve='ROC'),
                AUC(curve='PR')
              ]
            )


model.fit(
    x=X_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=epochs,
    validation_split=.1,
    callbacks=[tensorboard_callback, reduce_lr, early_stop]
    )

    

        