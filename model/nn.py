import tensorflow as tf
import sys
from keras import Model, layers
from keras import backend as K
from keras.layers import Conv1D, MaxPool1D, Dense, BatchNormalization
from keras.layers import Flatten, Layer, Concatenate, Reshape
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import *
from model.data import Dataset


class CalculateSimilarity(Layer):
  """Layer for calculating the similarity network

  Args:
      Layer (keras.layers.Layer): inherited
  """

  def __init__(self, sample_matrix, metric = 'euclidean', name=None, **kwargs):
    self.metric = metric
    self.trainable = False
    self.sample_matrix = tf.constant(value = sample_matrix, dtype=tf.float32)
    super(CalculateSimilarity, self).__init__(name=name, **kwargs)

  def call(self, inputs):
    print(inputs.shape)
    inputs = K.stack([inputs]*len(self.sample_matrix), axis=-2)
    temp = K.stack([self.sample_matrix]*len(inputs), axis=0)
    temp = tf.cast(temp, tf.float32)
    inputs = tf.cast(inputs, tf.float32)
    gamma = K.constant(value = 0.001)
    return K.exp(-gamma * K.sum(K.square(temp-inputs), axis=-1))


class multichannel_network(Model):
    def __init__(self,
                 feature_contained = ['cnv', 'methylation', 'mutation', 'gene_expression'],
                 dropout=.5,
                 n_channels = 4
                 ):
        super().__init__(self)
        self.dropout_rate = dropout
        self.ds = Dataset(feature_contained=feature_contained, response='AUC')
        self.ds.save()
        self.n_channels = n_channels

    def build(self, input_shape):

        # Molecular finger print, 881-dim sparse vector, Conv1D
        self.reshape_layer = Reshape(target_shape=(881,1))
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
        if 'cnv' in self.ds.feature_contained:
          self.similarity_layer_cnv = CalculateSimilarity(sample_matrix=self.ds.omics_data['cnv'])
          self.cnv_dense1 = layers.Dense(units=512, activation='relu')
          self.cnv_bn1 = layers.BatchNormalization()
          self.cnv_dense2 = layers.Dense(units=128, activation='relu')
          self.cnv_bn2 = layers.BatchNormalization()
          self.cnv_dropout = layers.Dropout(rate = self.dropout_rate)

        # Gene Expression
        if 'gene_expression' in self.ds.feature_contained:
          self.similarity_layer_expr = CalculateSimilarity(sample_matrix=self.ds.omics_data['gene_expression'])
          self.gene_expression_dense1 = layers.Dense(units=512, activation='relu')
          self.gene_expression_bn1 = layers.BatchNormalization()
          self.gene_expression_dense2 = layers.Dense(units=128, activation='relu')
          self.gene_expression_bn2 = layers.BatchNormalization()
          self.gene_expression_dropout = layers.Dropout(rate = self.dropout_rate)
        
        # Gene Mutations
        if 'mutation' in self.ds.feature_contained:
          self.similarity_layer_mut = CalculateSimilarity(sample_matrix=self.ds.omics_data['mutation'])
          self.mutations_dense1 = layers.Dense(units=512, activation='relu')
          self.mutations_bn1 = layers.BatchNormalization()
          self.mutations_dense2 = layers.Dense(units=128, activation='relu')
          self.mutations_bn2 = layers.BatchNormalization()
          self.mutations_dropout = layers.Dropout(rate = self.dropout_rate)

        # Methylation
        if 'methylation' in self.ds.feature_contained:
          self.similarity_layer_meth = CalculateSimilarity(sample_matrix=self.ds.omics_data['methylation'])
          self.methylation_dense1 = layers.Dense(units=512, activation='relu')
          self.methylation_bn1 = layers.BatchNormalization()
          self.methylation_dense2 = layers.Dense(units=128, activation='relu')
          self.methylation_bn2 = layers.BatchNormalization()
          self.methylation_dropout = layers.Dropout(rate = self.dropout_rate)

        # Integration Layer
        self.concat = Concatenate()
        self.integration_dense1 = Dense(512, activation='relu')
        self.intergration_bn1 = BatchNormalization()
        self.integration_dense2 = Dense(128, activation='relu')
        self.intergration_bn2 = BatchNormalization()
        self.integration_dense3 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        
        # Finger Print
        x = self.reshape_layer(inputs[0])
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

        feature = [x, r]
        # CNV
        if 'cnv' in self.ds.feature_contained:
          c = self.similarity_layer_cnv(inputs[2])
          c = self.cnv_dense1(c)
          c = self.cnv_bn1(c)
          c = self.cnv_dense2(c)
          c = self.cnv_bn2(c)
          c = self.cnv_dropout(c)
          feature.append(c)

        # gene_expression 
        if 'gene_expression' in self.ds.feature_contained:
          g = self.similarity_layer_expr(inputs[3])
          g = self.gene_expression_dense1(g)
          g = self.gene_expression_bn1(g)
          g = self.gene_expression_dense2(g)
          g = self.gene_expression_bn2(g)
          g = self.gene_expression_dropout(g)
          feature.append(g)

        # mutations
        if 'mutation' in self.ds.feature_contained:
          mut = self.similarity_layer_mut(inputs[4])
          mut = self.mutations_dense1(mut)
          mut = self.mutations_bn1(mut)
          mut = self.mutations_dense2(mut)
          mut = self.mutations_bn2(mut)
          mut = self.mutations_dropout(mut)
          feature.append(mut)

        # methylation
        if 'methylation' in self.ds.feature_contained:
          meth = self.similarity_layer_meth(inputs[5])
          meth = self.methylation_dense1(meth) 
          meth = self.methylation_bn1(meth)
          meth = self.methylation_dense2(meth)
          meth = self.methylation_bn2(meth)
          meth = self.methylation_dropout(meth)

        # Concat
        if len(feature) == 2:
           raise RuntimeError("No features of cellines specified")
        output = self.concat(feature)
        output = self.integration_dense1(output)
        output = self.intergration_bn1(output)
        output = self.integration_dense2(output)
        output = self.intergration_bn2(output)
        output = self.integration_dense3(output)

        return output

if __name__ == "__main__":
   _model = multichannel_network()
   from numpy.random import default_rng

   fingerprint_input_shape = (10, 881)
   rdkit2d_input_shape = (10, 200)
   cnv_input_shape = (10, _model.ds.cnv.shape[1])
   mutation_input_shape = (10, _model.ds.mutation.shape[1])
   gene_expression_input_shape = (10, _model.ds.fpkm.shape[1])
   methylation_input_shape = (10, _model.ds.methylation.shape[1])

   fingerprint = default_rng(42).random(fingerprint_input_shape)
   rdkit2d = default_rng(42).random(rdkit2d_input_shape)
   cnv = default_rng(42).random(cnv_input_shape)
   gene_expression = default_rng(42).random(gene_expression_input_shape)
   mutation = default_rng(42).random(mutation_input_shape)
   methylation = default_rng(42).random(methylation_input_shape)

   from keras.utils import plot_model

   output = _model([fingerprint, rdkit2d, cnv, gene_expression, mutation, methylation])
  
   _model.summary()
   print(output)