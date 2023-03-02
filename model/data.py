import pandas as pd
from os.path import join, realpath
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from sklearn import preprocessing, model_selection

# 注：min-Max归一化需要在分割完训练集和测试集和Validation set之后再进行

# Cached_data
cached_data = {}

# Path Define
cna_path = realpath("../../processed_data/cna.csv")
experiment_path = realpath("../../processed_data/experiment.csv")
fpkm_path = realpath("../../processed_data/fpkm.csv")
SMILES_path = realpath("../../processed_data/pubchem_id-SMILES.csv")

def load_data():
    cna = pd.read_csv(cna_path)
    experiment = pd.read_csv(experiment_path)
    fpkm = pd.read_csv(fpkm_path)
    preprocessing.minmax_scale(cna, feature_range=(0,1))



class Dataset:
    """DatasetClass

    Returns:
        _type_: _description_
    """
    # This class will facilitate the creation of Dataset
    def __init__(self, training=True):
        self.split = "training" if training else "test"
        self.cna = pd.read_csv(cna_path)
        self.experiment = pd.read_csv(experiment_path)
        self.fpkm = pd.read_csv(fpkm_path)
        self.scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=False)
        
        if selected_genes is not None:
            if type(selected_genes) == list:
                # list of genes
                selected_genes = selected_genes
            else:
                # file that will be used to load list of genes
                selected_genes_file = join(selected_genes_file, selected_genes)
                df = pd.read_csv(selected_genes_file, header=0)
                selected_genes = list(df['genes'])
        self.selected_genes = selected_genes

    def shape(self):
        return self._data.shape
    
    def prepare_response(self):
       pass 
        
    def statistics_and_describe(self):
        print("Copy Number Variation: ")
        print(self.cna)
        print("Experiment: ")
        print(self.experiment)
        print("Gene Expression FPKM: ")
        print(self.fpkm)

    def split_training_test_validation(self, test_size=0.1, 
                                       validation_size=.1, scale=True):
        model_selection.train_test_split()

    def return_raw_data(self, scale = True):
        pass

    def prepare_tf_dataset(self, split=None):
        features = tf.constant(self._data.values)
        labels = tf.constant(self.response.values)
        return tf.data.Dataset.from_tensor_slices((features, labels))
    
