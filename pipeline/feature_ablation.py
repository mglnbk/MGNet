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


def feature_ablation(feature, dataset="CTRP"):
    ds = Dataset(
        feature_contained=feature,
        dataset=dataset,
        set_label=True,
        response="AUC",
        threshold=0.58
    )
    result = []
    # Split train, test and validation set for training and testing
    partition = ds.k_fold(k=5)
    print(f"5-Fold Cross_Validation Begin!")

    for idx, sub in partition.items():
        print(f"This is {idx}...")
        train = sub['train']
        test = sub['test']

        train_generator = DataGenerator(sample_barcode=train, **ds.get_config(), batch_size=64)
        test_generator = DataGenerator(sample_barcode=test, **ds.get_config(), batch_size=64)

        model = multichannel_network(
            dataset=ds,
            train_sample_barcode=train,
            dropout=.5
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=
              [
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(curve='ROC'),
                AUC(curve='PR')
              ]
            )
        history = model.fit(x=train_generator, epochs=2)
        
        scores = model.evaluate(x=test_generator) 
        result.append(list(scores))
        print(result)
    
    feature_name = "_".join(feature)
    response = ds.target
    pd.DataFrame(result, columns=list(model.metrics_names)).to_csv(join(RESULT_PATH, f"Ablation_{feature_name}_{ds.dataset}_{response}.csv"), index=None)


if __name__ == "__main__":
    tf.random.set_seed(42)
    feature_ablation(['cna', 'gene_expression'])
