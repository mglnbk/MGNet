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

# ds = Dataset(['cnv', 'gene_expression', 'snv', 'methylation'])
# response = ds.response
# data = ds.return_feature()
# labels = response['AUC'].values
# scaler = MinMaxScaler()
# labels = labels.reshape(-1,1)
# scaler.fit(labels)
# labels = scaler.transform(labels)
from tensorflow import keras
from keras.layers import Conv1D, MaxPool1D, Layer
import numpy as np
from numpy.random import default_rng
from model.nn import multichannel_network

_model = multichannel_network()
fingerprint = []
rdkit2d = []
cnv = []
gene_expression = []
mutation = []
methylation = []
for i in _model.ds.processed_experiment['SAMPLE_BARCODE']:
    celline, cid = i.split("_")
    fingerprint.append(_model.ds.drug_info.drug_feature['fingerprint'].loc[int(cid)].values)
    rdkit2d.append(_model.ds.drug_info.drug_feature['rdkit2d'].loc[int(cid)].values)
    cnv.append(_model.ds.omics_data['cnv'].loc[celline].values)
    gene_expression.append(_model.ds.omics_data['gene_expression'].loc[celline].values)
    mutation.append(_model.ds.omics_data['snv'].loc[celline].values)
    methylation.append(_model.ds.omics_data['methylation'].loc[celline].values)


from sklearn.model_selection import train_test_split
response = _model.ds.response
labels = response['AUC'].values
y = []
for i in labels:
    if (i<=0.6):
        y.append(1)
    else:
        y.append(0)
labels = np.array(y[:10000])


batch_size = 64
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

_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
              loss="binary_crossentropy",
              metrics=
              [
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(curve='ROC'),
                AUC(curve='PR')
              ]
            )

fingerprint = np.array(fingerprint[:10000])
rdkit2d=np.array(rdkit2d[:10000])
cnv=np.array(cnv[:10000])
gene_expression = np.array(gene_expression[:10000])
mutation=np.array(mutation[:10000])
methylation=np.array(methylation[:10000])


_model.fit(
    x=[fingerprint, rdkit2d, cnv, gene_expression, mutation, methylation], 
    y=y, 
    batch_size=batch_size, 
    epochs=epochs,
    validation_split=.1,
    callbacks=[tensorboard_callback, reduce_lr, early_stop]
    )