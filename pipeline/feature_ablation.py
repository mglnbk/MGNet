import numpy as np
import tensorflow as tf
from keras.metrics import AUC
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.metrics import Precision, Recall
from keras.losses import BinaryCrossentropy
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
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


def feature_ablation(feature, dataset="CTRP"):
    result = []
# Dataset Setting:
# choose from ['methylation', 'gene_expression', 'cnv', 'mutation']
    # FEATURE = ['gene_expression', 'cnv', 'methylation', 'mutation']
    ds = Dataset(
        feature_contained=feature,
        dataset='CTRP',
        set_label=True,
        response='AUC',
        threshold=.58)
    ds.save()
    # CTRP, "AUC", 0.58, 0.001
    # GDSC, "AUC", .88, 0.001
    # model parameters settings
    lr_rate = 0.01
    dropout_rate = .5
    batch_size = 64
    epochs = 20

    # Split train, test and validation set for training and testing, build generators
    partition = ds.split(data=ds.sample_barcode, validation=True)
    train = partition['train']
    test = partition['test']
    validation = partition['validation']
    train_generator = DataGenerator(
        sample_barcode=train, **ds.get_config(), batch_size=batch_size)
    validation_generator = DataGenerator(
        sample_barcode=validation, **ds.get_config(), batch_size=batch_size)
    test_generator = DataGenerator(
        sample_barcode=test, **ds.get_config(), batch_size=batch_size)

    # Training parameters
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(
                                                        [ds.labels[x] for x in train]),
                                                    y=[ds.labels[x] for x in train])
    weights_dict = {i: w for i, w in enumerate(class_weights)}
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)


    def scheduler(epoch, lr):
        if (epoch % 5 == 0 and epoch != 0):
            return lr*0.1
        else:
            return lr


    reduce_lr = LearningRateScheduler(scheduler)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                               patience=5, min_lr=0.001)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)


    # model building
    model = multichannel_network(
        data=ds.omics_data,
        drug_data=ds.drug_info,
        feature_contained=ds.feature_contained,
        train_sample_barcode=train,
        dropout=dropout_rate
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate),
                loss=BinaryCrossentropy(),
                metrics=[
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(curve='ROC'),
        AUC(curve='PR')
    ]
    )

    history = model.fit(
        x=train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[reduce_lr, early_stop],
        class_weight=weights_dict
    )

    scores = model.evaluate(x=test_generator)
    result.append(list(scores))
    # save model
    pd.DataFrame(result, columns=list(model.metrics_names)).to_csv(f"Ablation_{feature}_{lr_rate}_{epochs}_{ds.target}.csv", index=None)

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

if __name__ == "__main__":
    feature_powerset = [list(i) for i in powerset(['cna', 'gene_expression', 'mutation', 'methylation']) if len(i)]
    for i in feature_powerset:
        print(i)
        feature_ablation(feature=i, dataset='CTRP')
