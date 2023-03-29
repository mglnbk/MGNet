from tensorflow import keras
import numpy as np
import sys
from keras import Model, layers
from keras.layers import Conv1D, MaxPool1D, Dense, BatchNormalization, Flatten, Layer, Concatenate
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
    print(inputs.shape, type(inputs))
    # gamma = 1/inputs.shape[1]
    # inputs_shape = (batch_size, n_genes or n_features)
    dist_matrix = pairwise_distances(X = np.array(inputs), 
                                     Y = self.sample_matrix.values,
                                     metric=self.metric)
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
        self.similarity_layer = CalculateSimilarity(sample_matrix=self.ds.omics_data['gene_expression'])
        self.gene_expression_dense1 = layers.Dense(units=512, activation='relu')
        self.gene_expression_bn1 = layers.BatchNormalization()
        self.gene_expression_dense2 = layers.Dense(units=128, activation='relu')
        self.gene_expression_bn2 = layers.BatchNormalization()
        self.gene_expression_dropout = layers.Dropout(rate = self.dropout_rate)

        # Methylation
        self.similarity_layer = CalculateSimilarity(sample_matrix=self.ds.omics_data['methylation'])
        self.methylation_dense1 = layers.Dense(units=512, activation='relu')
        self.methylation_bn1 = layers.BatchNormalization()
        self.methylation_dense2 = layers.Dense(units=128, activation='relu')
        self.methylation_bn2 = layers.BatchNormalization()
        self.methylation_dropout = layers.Dropout(rate = self.dropout_rate)

        # Gene Mutations
        self.similarity_layer = CalculateSimilarity(sample_matrix=self.ds.omics_data['snv'])
        self.mutations_dense1 = layers.Dense(units=512, activation='relu')
        self.mutations_bn1 = layers.BatchNormalization()
        self.mutations_dense2 = layers.Dense(units=128, activation='relu')
        self.mutations_bn2 = layers.BatchNormalization()
        self.mutations_dropout = layers.Dropout(rate = self.dropout_rate)

        # Integration Layer
        self.concat = Concatenate()
        self.integration_dense1 = Dense(512, activation='relu')
        self.intergration_bn1 = BatchNormalization()
        self.integration_dense2 = Dense(128, activation='relu')
        self.intergration_bn2 = BatchNormalization()
        self.integration_dense3 = Dense(1, activation='sigmoid')


    def call(self, inputs):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('Multi-channel must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        
        # Finger Print
        x = inputs[0]
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

        # Molecular 2D Rdkit
        r = inputs[1]
        r = self.rdkit_dense1(r)
        r = self.rdkit_bn1(r)
        r = self.rdkit_dense2(r)
        r = self.rdkit_bn2(r)
        r = self.rdkit_dropout(r)

        # CNV
        c = CalculateSimilarity(sample_matrix=self.ds.omics_data['cnv'])(inputs[2])
        c = self.cnv_dense1(c)
        c = self.cnv_bn1(c)
        c = self.cnv_dense2(c)
        c = self.cnv_bn2(c)
        c = self.cnv_dropout(c)

        # gene_expression 
        g = CalculateSimilarity(sample_matrix=self.ds.omics_data['gene_expression'])(inputs[3])
        g = self.gene_expression_dense1(g)
        g = self.gene_expression_bn1(g)
        g = self.gene_expression_dense2(g)
        g = self.gene_expression_bn2(g)
        g = self.gene_expression_dropout(g)

        # mutations
        mut = CalculateSimilarity(sample_matrix=self.ds.omics_data['snv'])(inputs[4])
        mut = self.mutations_dense1(mut)
        mut = self.mutations_bn1(mut)
        mut = self.mutations_dense2(mut)
        mut = self.mutations_bn2(mut)
        mut = self.mutations_dropout(mut)

        # methylation
        meth = CalculateSimilarity(sample_matrix=self.ds.omics_data['methylation'])(inputs[5])
        meth = self.methylation_dense1(meth) 
        meth = self.methylation_bn1(meth)
        meth = self.methylation_dense2(meth)
        meth = self.methylation_bn2(meth)
        meth = self.methylation_dropout(meth)

        # Concat
        output = self.concat([x, r, c, g, mut, meth])
        output = self.integration_dense1(output)
        output = self.intergration_bn1(output)
        output = self.integration_dense2(output)
        output = self.intergration_bn2(output)
        output = self.integration_dense3(output)

        return x

if __name__ == "__main__":
   _model = multichannel_network()
   from numpy.random import default_rng

   fingerprint_input_shape = (10, 881, 1)
   rdkit2d_input_shape = (10, 200)
   cnv_input_shape = (10, _model.ds.cnv.shape[1])
   mutation_input_shape = (10, _model.ds.snv.shape[1])
   gene_expression_input_shape = (10, _model.ds.fpkm.shape[1])
   methylation_input_shape = (10, _model.ds.methylation.shape[1])

   fingerprint = default_rng(42).random(fingerprint_input_shape)
   rdkit2d = default_rng(42).random(rdkit2d_input_shape)
   cnv = default_rng(42).random(cnv_input_shape)
   gene_expression = default_rng(42).random(gene_expression_input_shape)
   mutation = default_rng(42).random(mutation_input_shape)
   methylation = default_rng(42).random(methylation_input_shape)

   from keras.utils import plot_model
   _model.compile()
   output = _model([fingerprint, rdkit2d, cnv, gene_expression, mutation, methylation])

   _model.summary()