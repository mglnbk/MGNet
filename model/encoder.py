import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense


class encoder(layers.Layer):
  """encoder layer(Dense Layer, encode genetic profile data into
  condensed latent layer, stakced)

  Args:
    layers (tensorflow.layers.Layer): _description_
  """
  def __init__(self,  
               n_layers = 2
               ):
    super().__init__()
    self.n_layers=n_layers
    self.layers=[]
    
  def build(self, input_shape):
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.layers.append(Dense(units=128, activation="relu", input_shape=(last_dim,)))
    self.layers.append(Dense(units=64, activation="relu"))
    self.layers.append(Dense(units=32, activation="relu"))
    self.layers.append(Dense(units=4, activation='relu'))
    
  def call(self, inputs):
    outputs=inputs
    for layer in self.layers:
      outputs=layer(outputs)
    return outputs
