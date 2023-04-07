import tensorflow as tf
from keras.metrics import AUC
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.metrics import Precision, Recall
import datetime
from model.nn import multichannel_network
from model.data import Dataset, DataGenerator
from config_path import FEATURE

ds = Dataset(feature_contained=FEATURE)
model = multichannel_network(
    dataset=ds,
    dropout=.5
            )

batch_size = 64
epochs = 10


# Split train, test and validation set for training and testing
partition = model.ds.split(validation=True)
train = partition['train']
test = partition['test']
validation = partition['validation']


train_generator = DataGenerator(sample_barcode=train, **model.ds.get_config(), batch_size=batch_size)
validation_generator = DataGenerator(sample_barcode=validation, **model.ds.get_config(), batch_size=batch_size)
test_generator = DataGenerator(sample_barcode=test, **model.ds.get_config(), batch_size=batch_size)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def scheduler(epoch, lr):
    if(epoch % 5 ==0 and epoch !=0):
        return lr*0.1
    else:
        return lr
reduce_lr = LearningRateScheduler(scheduler)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience=10)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=
              [
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(curve='ROC'),
                AUC(curve='PR')
              ]
            )

history = model.fit(x=train_generator, 
                    epochs=epochs,
                    validation_data=validation_generator, 
                    callbacks=[reduce_lr, early_stop]
                    )

model.evaluate(x=test_generator) 

