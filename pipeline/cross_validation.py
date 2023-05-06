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
ds.save()
# training parameters settings
lr_rate = 1e-2
dropout_rate = .5
batch_size = 64
epochs = 15

# log parameters

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
def scheduler(epoch, lr):
    if(epoch % 5 ==0 and epoch !=0):
        return lr*0.1
    else:
        return lr
reduce_lr = LearningRateScheduler(scheduler)
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
        
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique([ds.labels[x] for x in train]),
                                                 y=[ds.labels[x] for x in train])
        weights_dict = {i:w for i,w in enumerate(class_weights)}

        model = multichannel_network(
            data=ds.omics_data,
            drug_data=ds.drug_info,
            feature_contained=['gene_expression', 'cnv', 'methylation', 'mutation'],
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
                    callbacks=[reduce_lr, tensorboard_callback],
                    class_weight = weights_dict
                    )
        
        scores = model.evaluate(x=test_generator) 
        result.append(list(scores))
        print(result)
        dataset_name = ds.dataset
        tf.saved_model.save(model, f'{dataset_name}_{lr_rate}_5-0.1_{batch_size}_{epochs}_all')
    
    feature_name = "_".join(FEATURE)
    response = ds.target
    pd.DataFrame(result, columns=list(model.metrics_names)).to_csv(join(RESULT_PATH, f"CV_{feature_name}_{lr_rate}_{epochs}_{response}.csv"), index=None)

if __name__ == "__main__":
    cross_validation()
