from tensorflow import keras
import numpy as np
import sys
from keras import Model, layers
from keras.layers import Conv1D, MaxPool1D, Dense, BatchNormalization, Flatten, Layer, Concatenate
from keras.utils import plot_model
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import *
from model.data import Dataset
from sklearn.metrics import pairwise_distances

class CalculateSimilarity(Layer):

  def __init__(self, sample_matrix, metric = 'euclidean', name=None, **kwargs):
    self.metric = metric 
    self.trainable = False
    self.sample_matrix = sample_matrix
    super(CalculateSimilarity, self).__init__(name=name, **kwargs)

  def call(self, inputs):
    # gamma = 1/inputs.shape[1]
    # inputs_shape = (batch_size, n_genes) or (1, n_genes)
    dist_matrix = pairwise_distances(X = inputs, Y = self.sample_matrix, metric=self.metric)
    gamma = 0.001
    return np.exp(-np.square(dist_matrix) * gamma)

  def compute_output_shape(self, input_shape):
    return input_shape



class multichannel_network(Model):
    def __init__(self, 
                 feature_contained = ['cnv', 
                                      'gene_expression', 
                                      'snv', 
                                      'methylation'
                                      ],
                 dropout=.5,
                 n_channels = 4
                 ):
        super().__init__(self)
        self.dropout_rate = dropout
        self.ds = Dataset(feature_contained, 'GDSC')
        self.n_channels = n_channels
        
    def build(self, input_shape):

        # Molecular finger print, 881-dim sparse vector, Conv1D
        self.fp_conv1 = Conv1D(filters=4, kernel_size=8, activation='relu')
        self.fp_bn1 = BatchNormalization()
        self.fp_pool1 = MaxPool1D(3, 3)
        self.fp_conv2 = Conv1D(filters=self.n_channels * 2, kernel_size=8, activation='relu')
        self.fp_bn2 = BatchNormalization()
        self.fp_pool2 = MaxPool1D(3, 3)
        self.fp_conv3 = Conv1D(filters=self.n_channels * 4, kernel_size=8, activation='relu')
        self.fp_bn3 = BatchNormalization()
        self.fp_pool3 = MaxPool1D(3, 3)
        self.flatten = Flatten()
        self.fp_fc1 = Dense(512)
        self.fp_bn4 = BatchNormalization()
        self.fp_fc2 = Dense(128)
        self.fp_bn5 = BatchNormalization()


        # Molecular 2D Component, 200-dim vector
        self.rdkit_dense1 = layers.Dense(units=512, activation='relu')
        self.rdkit_bn1 = layers.BatchNormalization()
        self.rdkit_dense2 = layers.Dense(units=128, activation='relu')
        self.rdkit_bn2 = layers.BatchNormalization()
        self.rdkit_dropout = layers.Dropout(rate = self.dropout_rate)


        # Copy Number Variation
        self.similarity_layer = CalculateSimilarity(sample_matrix=self.ds.omics_data['cnv'])
        self.cnv_dense1 = layers.Dense(units=512, activation='relu')
        self.cnv_bn1 = layers.BatchNormalization()
        self.cnv_dense2 = layers.Dense(units=128, activation='relu')
        self.cnv_bn2 = layers.BatchNormalization()
        self.cnv_dropout = layers.Dropout(rate = self.dropout_rate)
        

        # Gene Expression



        # Methylation

        
        # Gene Mutations


        # Integration Layer
        self.concat = Concatenate()
        self.integration_dense1 = Dense(512, activation='relu')
        self.intergration_bn1 = BatchNormalization()
        self.integration_dense2 = Dense(128, activation='relu')
        self.intergration_bn2 = BatchNormalization()
        self.integration_dense3 = Dense(1, activation='sigmoid')


    def call(self, data):

        # Finger Print
        x = data['fingerprint']
        x = self.fp_conv1(x)
        x = self.fp_bn1(x)
        x = self.fp_pool1(x)
        x = self.fp_conv2(x)
        x = self.fp_bn2(x)
        x = self.fp_pool2(x)
        x = self.fp_conv3(x)
        x = self.fp_bn3(x)
        x = self.fp_pool3(x)
        x = self.flatten(x)
        x = self.fp_fc1(x)
        x = self.fp_bn4(x)
        x = self.fp_fc2(x)
        x = self.fp_bn5(x)

        #
        
        return x
    
