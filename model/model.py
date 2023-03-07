import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras import layers
from keras import backend
from keras import Sequential
from encoder import encoder

class multichannel_network(Model):
    def __init__(self, nb_filter=3, input_feature=12, dropout=.5):
        super().__init__(self)
        # Gene Exression Filter
        


        # Copy Number Variation Filter



        # Molecular Filter



        # Integration Filter

        
    def build(self, input_shape):
        return super().build(input_shape)
        

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)
        





