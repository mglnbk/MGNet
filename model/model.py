import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend
from keras import Sequential
from encoder import encoder

class multichannel_network(tf.keras.Model):
    def __init__(self, nb_filter=3, input_feature=12, dropout=.5):
        super().__init__(self)
        # Gene Exression Filter
            


        # Copy Number Variation Filter



        # Molecular Filter



        # Integration Filter




if __name__ == "__main__":
    gnet = gene_net()
    gnet(tf.ones(shape=(1011, 10)))
    gnet.summary()
