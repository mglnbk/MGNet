import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import Model, layers
from data import Dataset
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import * 
import subprocess
from beta_vae import *

class multichannel_network(Model):
    def __init__(self, 
                 feature_contained = ['cnv', 'gene_expression', 
                                      'snv', 'methylation'],
                 dropout=.5
                 ):
        super().__init__(self)
        self.dropout_rate = dropout
        self.ds = Dataset(feature_contained)
        self.feature = self.ds.return_feature()
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(units=1024, activation='relu'),
        self.bn1 = layers.BatchNormalization(),
        self.dense2 = layers.Dense(units=128, activation='relu'),
        self.bn2 = layers.BatchNormalization(),
        self.dropout = layers.Dropout(rate = self.dropout_rate),
        self.dense3 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False, mask=None):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        if training:
            x = self.dropout(x)
        x = self.dense3(x)
        return x
