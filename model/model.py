import tensorflow as tf
from tensorflow import keras
import sys
from keras import Model, layers
from data import Dataset
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import * 
from model.data import Dataset


class multichannel_network(Model):
    def __init__(self, 
                 feature_contained = ['cnv', 'gene_expression', 
                                      'snv', 'methylation'],
                 dropout=.5
                 ):
        super().__init__(self)
        self.dropout_rate = dropout
        self.ds = Dataset(feature_contained, 'GDSC')
        
    def build(self, input_shape):
        # Molecular finger print, Conv1D
        

        # Molecular 2D Component
        self.fp_dense1 = layers.Dense(units=1024, activation='relu')
        self.fp_bn1 = layers.BatchNormalization()
        self.fp_dense2 = layers.Dense(units=256, activation='relu')
        self.fp_bn2 = layers.BatchNormalization()
        self.fp_dropout = layers.Dropout(rate = self.dropout_rate)

        # 

    def call(self, data):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        if training:
            x = self.dropout(x)
        x = self.dense3(x)
        return x
