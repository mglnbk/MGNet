import tensorflow as tf
import sys
from keras import Model, layers, regularizers
from keras import backend as K
from keras.layers import Conv1D, MaxPool1D, Dense, BatchNormalization
from keras.layers import Flatten, Layer, Concatenate, Reshape
from keras.layers import Normalization
from pathlib import Path
from os.path import realpath
from sklearn.preprocessing import StandardScaler
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import *


class CalculateSimilarity(Layer):
  """Layer for calculating the similarity network

  Args:
      Layer (keras.layers.Layer): inherited
  """

  def __init__(self, sample_matrix, trainable=False, metric = 'euclidean', name=None, **kwargs):
    super(CalculateSimilarity, self).__init__(name=name, trainable=trainable, **kwargs)
    self.metric = metric
    self.trainable = trainable
    self.sample_matrix = K.constant(value = sample_matrix, dtype=tf.float32, name="Similarity_Matrix")

  def get_config(self):
    config = super().get_config().copy()
    config.update({
       'metric': self.metric,
       'trainable': self.trainable,
       'sample_matrix': self.sample_matrix
    })
    return config

  def call(self, inputs):
    inputs = K.stack([inputs]*self.sample_matrix.shape[0], axis=-2) # b * (n) *f
    inputs = K.cast(inputs, tf.float32)
    gamma = K.constant(value = 0.001) # batch * feature
    inputs = K.exp(-gamma * K.sum(K.square(inputs-self.sample_matrix), axis=-1)) # (b) * n
    return inputs

class multichannel_network(Model):
    def __init__(self,
                 data,
                 drug_data,
                 feature_contained,
                 train_sample_barcode,
                 dropout=.5
                 ):
        super().__init__(self)
        self.dropout_rate = dropout
        self.data = data
        self.drug_data = drug_data
        self.feature_contained = feature_contained
        self.cell_line = list(set(i.split("_")[0] for i in train_sample_barcode))
        self.drug = list(set(i.split("_")[1] for i in train_sample_barcode)) 
        self.regularize = {"kernel_regularizer": regularizers.l2(0.001),
                           "bias_regularizer": regularizers.l2(0.001)}
        self.encoded_feature = []

        # Model Layers Constructor        
        # Molecular finger print, 881-dim sparse vector, Conv1D
        self.fp_norm = Normalization(axis=-1)
        self.fp_norm.adapt(data=self.drug_data.drug_feature['fingerprint'].loc[self.drug].values)
        self.reshape_layer = Reshape(target_shape=(881,1))
        self.fp_conv1 = Conv1D(filters=4, 
                               kernel_size=8, 
                               activation='relu', 
                               **self.regularize
                               )
        self.fp_bn1 = BatchNormalization()
        self.fp_pool1 = MaxPool1D(3, 3)
        self.fp_conv2 = Conv1D(filters=8, 
                               kernel_size=8, 
                               activation='relu', 
                               **self.regularize)
        self.fp_bn2 = BatchNormalization()
        self.fp_pool2 = MaxPool1D(3, 3)
        self.fp_conv3 = Conv1D(filters=16, 
                               kernel_size=8, 
                               activation='relu', 
                               **self.regularize)
        self.fp_bn3 = BatchNormalization()
        self.fp_pool3 = MaxPool1D(3, 3)
        self.flatten = Flatten()
        self.fp_fc1 = Dense(512, activation='relu', **self.regularize)
        self.fp_bn4 = BatchNormalization()
        self.fp_fc2 = Dense(128, activation=None)
        self.fp_bn5 = BatchNormalization()
        self.fp_dropout = layers.Dropout(rate=self.dropout_rate)

        # Molecular 2D Component, 200-dim vector
        self.rdkit_norm = Normalization(axis=-1)
        self.rdkit_norm.adapt(data=self.drug_data.drug_feature['rdkit2d'].loc[self.drug].values)
        self.rdkit_dense1 = layers.Dense(units=512,
                                        activation='relu',
                                        **self.regularize)
        self.rdkit_bn1 = layers.BatchNormalization()
        self.rdkit_dense2 = layers.Dense(units=128, activation=None)
        self.rdkit_bn2 = layers.BatchNormalization()
        self.rdkit_dropout = layers.Dropout(rate = self.dropout_rate)

        # Copy Number Variation
        if 'cnv' in self.feature_contained:
          self.cnv_norm = Normalization(axis=-1)
          self.cnv_norm.adapt(data=self.data['cnv'].loc[self.cell_line].values)
          self.cnv_scaler = StandardScaler()
          self.similarity_layer_cnv = CalculateSimilarity(
            sample_matrix=self.cnv_scaler.fit_transform(self.data['cnv'].loc[self.cell_line])
                                                          )
          self.cnv_dense1 = layers.Dense(units=512, activation='relu', **self.regularize)
          self.cnv_bn1 = layers.BatchNormalization()
          self.cnv_dense2 = layers.Dense(units=128, activation=None)
          self.cnv_bn2 = layers.BatchNormalization()
          self.cnv_dropout = layers.Dropout(rate = self.dropout_rate)

        # Gene Expression
        if 'gene_expression' in self.feature_contained:
          self.gene_expression_norm = Normalization(axis=-1)
          self.gene_expression_norm.adapt(data=self.data['gene_expression'].loc[self.cell_line].values)

          self.gene_expression_scaler = StandardScaler()
          self.similarity_layer_expr = CalculateSimilarity(
            sample_matrix=self.gene_expression_scaler.fit_transform(self.data['gene_expression'].loc[self.cell_line])
            )
          self.gene_expression_dense1 = layers.Dense(units=512, activation='relu', **self.regularize)
          self.gene_expression_bn1 = layers.BatchNormalization()
          self.gene_expression_dense2 = layers.Dense(units=128, activation=None)
          self.gene_expression_bn2 = layers.BatchNormalization()
          self.gene_expression_dropout = layers.Dropout(rate = self.dropout_rate)
        
        # Gene Mutations
        if 'mutation' in self.feature_contained:
          self.mutation_norm = Normalization(axis=-1)
          self.mutation_norm.adapt(data=self.data['mutation'].loc[self.cell_line].values)

          self.mutation_scaler = StandardScaler()
          self.similarity_layer_mut = CalculateSimilarity(
            sample_matrix=self.mutation_scaler.fit_transform(self.data['mutation'].loc[self.cell_line])
            )
          self.mutations_dense1 = layers.Dense(units=512, activation='relu', **self.regularize)
          self.mutations_bn1 = layers.BatchNormalization()
          self.mutations_dense2 = layers.Dense(units=128, activation=None)
          self.mutations_bn2 = layers.BatchNormalization()
          self.mutations_dropout = layers.Dropout(rate = self.dropout_rate)

        # Methylation
        if 'methylation' in self.feature_contained:
          self.methylation_norm = Normalization(axis=-1)
          self.methylation_norm.adapt(data=self.data['methylation'].loc[self.cell_line].values)
          self.methylation_scaler = StandardScaler()
          self.similarity_layer_meth = CalculateSimilarity(
            sample_matrix=self.methylation_scaler.fit_transform(self.data['methylation'].loc[self.cell_line])
            )
          self.methylation_dense1 = layers.Dense(units=512, activation='relu', **self.regularize)
          self.methylation_bn1 = layers.BatchNormalization()
          self.methylation_dense2 = layers.Dense(units=128, activation=None)
          self.methylation_bn2 = layers.BatchNormalization()
          self.methylation_dropout = layers.Dropout(rate = self.dropout_rate)

        # Integration Layer
        self.concat = Concatenate()
        self.integration_dense1 = Dense(512, activation='relu')
        self.intergration_bn1 = BatchNormalization()
        self.integration_dense2 = Dense(128, activation='relu')
        self.intergration_bn2 = BatchNormalization()
        self.integration_dense3 = Dense(1, activation='sigmoid')

    def get_config(self):
      return super().get_config()

    def call(self, inputs, training=None):
        
        # Finger Print
        x = inputs['fingerprint']
        x = self.fp_norm(x)
        x = self.reshape_layer(x)
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
        if training:
          x = self.fp_dropout(x)

        # Molecular 2D Rdkit
        r = inputs['rdkit2d']
        r = self.rdkit_norm(r)
        r = self.rdkit_dense1(r)
        r = self.rdkit_bn1(r)
        r = self.rdkit_dense2(r)
        r = self.rdkit_bn2(r)
        if training:
          r = self.rdkit_dropout(r)

        feature = [x, r]
        # CNV
        if 'cnv' in self.feature_contained:
          c = inputs['cnv']
          c = self.cnv_norm(c)
          c = self.similarity_layer_cnv(c)
          c = self.cnv_dense1(c)
          c = self.cnv_bn1(c)
          c = self.cnv_dense2(c)
          c = self.cnv_bn2(c)
          if training:
            c = self.cnv_dropout(c)
          feature.append(c)

        # gene_expression 
        if 'gene_expression' in self.feature_contained:
          g = inputs['gene_expression']
          g = self.gene_expression_norm(g)
          g = self.similarity_layer_expr(g)
          g = self.gene_expression_dense1(g)
          g = self.gene_expression_bn1(g)
          g = self.gene_expression_dense2(g)
          g = self.gene_expression_bn2(g)
          if training:
            g = self.gene_expression_dropout(g)
          feature.append(g)

        # mutations
        if 'mutation' in self.feature_contained:
          mut = inputs['mutation']
          mut = self.mutation_norm(mut)
          mut = self.similarity_layer_mut(mut)
          mut = self.mutations_dense1(mut)
          mut = self.mutations_bn1(mut)
          mut = self.mutations_dense2(mut)
          mut = self.mutations_bn2(mut)
          if training:
            mut = self.mutations_dropout(mut)
          feature.append(mut)

        # methylation
        if 'methylation' in self.feature_contained:
          meth = inputs['methylation']
          meth = self.methylation_norm(meth)
          meth = self.similarity_layer_meth(meth)
          meth = self.methylation_dense1(meth) 
          meth = self.methylation_bn1(meth)
          meth = self.methylation_dense2(meth)
          meth = self.methylation_bn2(meth)
          if training:
            meth = self.methylation_dropout(meth)
          feature.append(meth)

        # Concat
        if len(feature) == 2:
           raise RuntimeError("No features of cellines specified")
        output = self.concat(feature)
        if not training:
          self.encoded_feature.append(output)
        output = self.integration_dense1(output)
        output = self.intergration_bn1(output)
        output = self.integration_dense2(output)
        output = self.intergration_bn2(output)
        output = self.integration_dense3(output)
        return output
