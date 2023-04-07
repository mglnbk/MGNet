from sklearn.model_selection import cross_val_score, StratifiedKFold
import sys
from pathlib import Path
from os.path import realpath, join
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import *
from model.data import Dataset
import tensorflow as tf
from keras.metrics import AUC
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.metrics import Precision, Recall
import datetime
from model.nn import multichannel_network
from model.data import Dataset, DataGenerator
import pandas as pd

# model parameters settings
lr_rate = 1e-3
dropout_rate = .5

# choose from ['methylation', 'gene_expression', 'cnv', 'mutation']
FEATURE = ['gene_expression', 'cnv', 'methylation', 'mutation']

ds = Dataset(
    feature_contained=FEATURE, 
    dataset='GDSC', 
    set_label=True, 
    response='', 
    threshold=.4
    )

batch_size = 64
epochs = 10

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
    model = multichannel_network(
        dataset=ds,
        dropout=.5
    )
    result = []
    # Split train, test and validation set for training and testing
    partition = ds.k_fold(k=k_fold)
    print(f"{k_fold}-Fold Cross_Validation Begin!")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=
              [
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(curve='ROC'),
                AUC(curve='PR')
              ]
            )

    for idx, sub in partition.items():
        print(f"This is {idx}...")
        train = sub['train']
        test = sub['test']

        train_generator = DataGenerator(sample_barcode=train, **ds.get_config(), batch_size=batch_size)
        test_generator = DataGenerator(sample_barcode=test, **ds.get_config(), batch_size=batch_size)

        log_dir = "logs/cv_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(x=train_generator, 
                    epochs=epochs,
                    validation_data=test_generator, 
                    callbacks=[reduce_lr, early_stop, tensorboard_callback]
                    )
        
        scores = model.evaluate(x=test_generator) 
        result.append(list(scores))
    
    pd.DataFrame(result, columns=list(model.metrics_names)).to_csv(join(RESULT_PATH, "CV.csv"), index=None)

if __name__ == "__main__":
    cross_validation()
