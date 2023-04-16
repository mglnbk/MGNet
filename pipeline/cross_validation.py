import tensorflow as tf
from keras.metrics import AUC
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.metrics import Precision, Recall
import datetime
import sys
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import *
from model.nn import multichannel_network
from model.data import Dataset, DataGenerator
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from os.path import join

# Dataset Setting: 
## choose from ['methylation', 'gene_expression', 'cnv', 'mutation']
FEATURE = ['gene_expression', 'cnv', 'methylation', 'mutation']
ds = Dataset(
    feature_contained=FEATURE, 
    dataset='CTRP', 
    set_label=True, 
    response='AUC', 
    threshold=.58)

# model parameters settings
lr_rate = 1e-3
dropout_rate = .5
batch_size = 64
epochs = 10

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

def cross_validation(k_fold=5):
    result = []
    # Split train, test and validation set for training and testing
    partition = ds.k_fold(k=k_fold)
    print(f"{k_fold}-Fold Cross_Validation Begin!")

    for idx, sub in partition.items():
        print(f"This is {idx}...")
        train = sub['train']
        test = sub['test']

        train_generator = DataGenerator(sample_barcode=train, **ds.get_config(), batch_size=batch_size)
        test_generator = DataGenerator(sample_barcode=test, **ds.get_config(), batch_size=batch_size)

        model = multichannel_network(
            dataset=ds,
            train_sample_barcode=train,
            dropout=.5
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=
              [
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(curve='ROC'),
                AUC(curve='PR')
              ]
            )
        log_dir = "logs/cv_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(x=train_generator, 
                    epochs=epochs,
                    #validation_data=test_generator, 
                    callbacks=[reduce_lr, early_stop, tensorboard_callback],
                    class_weight = weights_dict
                    )
        
        scores = model.evaluate(x=test_generator) 
        result.append(list(scores))
        print(result)
        dataset_name = ds.dataset
        model.save(filepath=join(RESULT_PATH, f"{idx}_{k_fold}_{dataset_name}_fold_model"), save_format='tf')
    
    feature_name = "_".join(FEATURE)
    response = ds.target
    pd.DataFrame(result, columns=list(model.metrics_names)).to_csv(join(RESULT_PATH, f"CV_{feature_name}_{lr_rate}_{epochs}_{response}.csv"), index=None)

if __name__ == "__main__":
    cross_validation()
