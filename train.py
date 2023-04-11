import tensorflow as tf
from keras.metrics import AUC
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.metrics import Precision, Recall
from keras.losses import BinaryCrossentropy
import datetime
from model.nn import multichannel_network
from model.data import Dataset, DataGenerator
from sklearn.utils import class_weight
import numpy as np

# Dataset Setting: 
## choose from ['methylation', 'gene_expression', 'cnv', 'mutation']
FEATURE = ['gene_expression', 'cnv', 'methylation', 'mutation']
ds = Dataset(
    feature_contained=FEATURE, 
    dataset='GDSC', 
    set_label=True, 
    response='AUC', 
    threshold=.88)

# model parameters settings
lr_rate = 3e-4
dropout_rate = .5
batch_size = 32
epochs = 2

# Split train, test and validation set for training and testing, build generators
partition = ds.split(validation=True)
train = partition['train']
test = partition['test']
validation = partition['validation']
train_generator = DataGenerator(sample_barcode=train, **ds.get_config(), batch_size=batch_size)
validation_generator = DataGenerator(sample_barcode=validation, **ds.get_config(), batch_size=batch_size)
test_generator = DataGenerator(sample_barcode=test, **ds.get_config(), batch_size=batch_size)

# Training parameters
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique([ds.labels[x] for x in train]),
                                                 y=[ds.labels[x] for x in train])
weights_dict = {i:w for i,w in enumerate(class_weights)}
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


# model building
model = multichannel_network(
    dataset=ds,
    dropout=dropout_rate
    )

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate),
              loss=BinaryCrossentropy(),
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
                    callbacks=[reduce_lr, early_stop],
                    class_weight=weights_dict
                    )

model.evaluate(x=test_generator) 


# save model
model.save("model.h5")


