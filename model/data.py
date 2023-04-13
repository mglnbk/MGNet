import pandas as pd
import numpy as np
import keras
import sys
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import *
from model.drug import Drug
from os.path import join
from sklearn.model_selection import train_test_split, StratifiedKFold
import h5py

# 注：min-Max归一化需要在分割完训练集和测试集和Validation set之后再进行

def load_data_cna(data_path) -> pd.DataFrame:
    print("Loading Copy Number Abberation Data...")
    all_cna = pd.read_csv(data_path, low_memory=False,
                    skiprows=lambda x: x in [0, 2])
    all_cna = all_cna.drop(columns=['model_name'])
    all_cna.rename(columns={"Unnamed: 1": "celline_barcode"}, inplace=True)
    all_cna.set_index('celline_barcode', inplace=True)
    all_cna = all_cna.T
    all_cna.fillna(0.0, inplace=True)
    print("Loading Copy Number Abberation Data Done")
    
    return all_cna

def load_data_celline(celline_data_path, data_source) -> pd.DataFrame:
    print("Loading Celline Data...")
    if data_source == "GDSC":
        celline = pd.read_excel(celline_data_path, sheet_name=0)
    print("Loading Celline Data Done")
    return celline

def load_data_fpkm(data_path) -> pd.DataFrame:
    print("Loading Gene Expression Data...")
    all_fpkm = pd.read_csv(data_path, low_memory=False,
                    skiprows=lambda x: x in [0, 2, 3, 4])
    all_fpkm = all_fpkm.drop(columns=['model_name'])
    all_fpkm.rename(columns={"Unnamed: 1": "celline_barcode"}, inplace=True)
    all_fpkm.set_index('celline_barcode', inplace=True)
    all_fpkm = all_fpkm.T
    all_fpkm.fillna(0.0, inplace=True)
    print("Loading Gene Expression Data Done")
    return all_fpkm

def load_data_experiment(data_path, data_source) -> pd.DataFrame:
    
    if data_source == "GDSC":
        print("Loading GDSC Experiment Data...")
        experiment = pd.read_excel(data_path)
        experiment = experiment[['DATASET', 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50', 'AUC']]
    if data_source == "CTRP":
        print("Loading CTRP Experiment Data...")
        experiment = pd.read_csv(RAW_EXPERIMENT_CTRP_PATH, sep='\t')
        meta_celline = pd.read_csv(RAW_CELLINE_CTRP_PATH, sep="\t")
        meta_experiment = pd.read_csv(RAW_META_EXPERIMENT_CTRP_PATH, sep='\t')
        meta_compound = pd.read_csv(RAW_COMPOUND_CTRP_PATH, sep='\t')
        meta_celline = meta_celline[['master_ccl_id', 'ccl_name']]
        meta_compound = meta_compound[['master_cpd_id', 'cpd_name', 'cpd_smiles']]
        meta_experiment = meta_experiment[['experiment_id','master_ccl_id']]
        experiment = experiment[['experiment_id', 'area_under_curve', 'master_cpd_id']]
        experiment = experiment.join(meta_experiment.set_index('experiment_id'), on='experiment_id',how='inner')
        experiment = experiment.join(meta_compound.set_index('master_cpd_id'), on='master_cpd_id',how='inner')
        experiment = experiment.join(meta_celline.set_index('master_ccl_id'), on='master_ccl_id', how='inner')
        experiment = experiment[['cpd_name', 
                                 'cpd_smiles', 'ccl_name', 
                                 'area_under_curve'
                                 ]]
        experiment['DATASET'] = ['CTRP']*len(experiment)
        experiment.rename(columns={"cpd_name": "DRUG_NAME", "ccl_name":"CELL_LINE_NAME",
                           "cpd_smiles": "SMILES", "area_under_curve": "AUC"}, inplace=True)
    print("Loading Experiment Data Done")
    return experiment

def load_data_methylation(data_path) -> pd.DataFrame:
    print("Loading Methylation Data...")
    met_df = pd.read_csv(data_path, sep = '\t', index_col=0)
    temp = pd.read_excel(RAW_SENTRIX2SAMPLE_GDSC_PATH, sheet_name=0)
    temp['Sentrix_Barcode'] = list("_".join([i,j]) for i, j in zip(temp['Sentrix_ID'].astype(str), temp['Sentrix_Position']))
    tb = dict(zip(temp['Sentrix_Barcode'], temp['Sample_Name']))
    met_df.rename(columns=tb, inplace=True)
    met_df = met_df.T
    met_df = met_df.reset_index().drop_duplicates(subset='index', keep='first').rename(columns={'index': 'celline'})
    met_df = met_df.set_index('celline')
    print("Loading Methylation Data Done")
    return met_df

def load_data_mutation(data_path) -> pd.DataFrame:
    print("Loading Mutations Data...")
    mutation_df = pd.read_csv(data_path, low_memory=False)
    important_only = ['cds_disrupted','complex_sub','downstream', 'ess_splice','frameshift','missense','nonsense','silent','splice_region','start_lost','stop_lost','upstream']
    mutation_df = mutation_df[mutation_df['effect'].isin(important_only)]
    df_table = pd.pivot_table(data=mutation_df, 
                                index='model_name', 
                                columns='gene_symbol', 
                                values='effect',
                                aggfunc='count',
                                fill_value=0)
    print("Loading Mutations Data Done")
    return df_table


class Dataset():
    """Dataset, include preprocessing and pre-split operations and etc.

    Returns:
        _type_: _description_
    """
    # This class will facilitate the creation of Dataset
    def __init__(self, 
                 feature_contained = ['cnv', 'gene_expression'], 
                 dataset = 'GDSC',
                 set_label = True,
                 response = 'AUC', # 'LN_IC50'
                 threshold = 0.4
                 ):
        self.feature_contained = feature_contained
        self.dataset = dataset
        self.target = response
        self.threshold = threshold

        # load multi-omics data
        if "cnv" in feature_contained:
            self.cnv = load_data_cna(RAW_CNV_GDSC_PATH)
        if "gene_expression" in feature_contained:
            self.fpkm = load_data_fpkm(RAW_FPKM_GDSC_PATH)
        if "methylation" in feature_contained:
            self.methylation = load_data_methylation(RAW_METHYLATION_GDSC_PATH)
        if "mutation" in feature_contained:
            self.mutation = load_data_mutation(RAW_SNV_GDSC_PATH)

        self.drug_info = Drug(method='manual')
        # self.celline = load_data_celline(RAW_CELLINE_GDSC_PATH, "GDSC")
        self.experiment = load_data_experiment(RAW_EXPERIMENT_GDSC_PATH, self.dataset)
        # First to preprocess experiment data matrix !
        self.celline_barcode = self.get_celline_barcode()
        self.processed_experiment = self.preprocess_experiment()

        # Then preprocess omics data and response
        self.omics_data = self.preprocess_omics()
        self.response = self.prepare_response(res=self.target) # pd.DataFrame

        # Prepare keras dataset configurations, for training
        if set_label:
            y = []
            for i in self.processed_experiment[self.target]:
                if (i<=self.threshold):
                    y.append(1)
                else:
                    y.append(0)
            self.labels = {
                _id:_y for _, (_id, _y) in enumerate(zip(self.response['SAMPLE_BARCODE'], y))
            }
        else:
            self.labels = {
                _id:_y for _, (_id, _y) in enumerate(zip(self.response['SAMPLE_BARCODE'], self.processed_experiment[self.target]))
            }
        self.sample_barcode = list(self.response['SAMPLE_BARCODE'])
    
    def split(self, rate=0.1, validation=True, seed = 42):
        """split train and test(or validation) dataset

        Args:
            rate (float, optional): test_size. Defaults to 0.1.
            validation (bool, optional): val or not. Defaults to True.

        Returns:
            dict: sample_barcode dict named partition['train', 'test', 'validation']
        """
        if validation==True:
            sample_train, sample_test = train_test_split(self.sample_barcode,
                                                         test_size=rate*2,
                                                         random_state=seed)
            sample_test, sample_validation = train_test_split(sample_test,
                                                              test_size=.5,
                                                              random_state=seed)
            partition = {
                "train": sample_train,
                "test": sample_test,
                "validation": sample_validation
            }
            return partition
        else:
            sample_train, sample_test = train_test_split(self.sample_barcode,
                                                         test_size=rate,
                                                         random_state=seed)
            partition = {
                "train": sample_train,
                "test": sample_test,
            }
            return partition
        
    def k_fold(self, k, seed=42):
        """return k-fold sample_barcode list, Note that k cant be
            greater than number of samples in each class

        Args:
            k (int): k fold
            seed (int, optional): _description_. Defaults to 42.

        Returns:
            dict: {
                fold_1: {train: ['12', ...], test: ['RCV12', ...]},
                fold_2: ... 
            }
        """
        skf = StratifiedKFold(n_splits=k)
        partition = {}
        for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(self.labels)), list(self.labels.values()))):
            partition[f'fold_{i}'] = {
                "train": list(self.response.iloc[train_index]['SAMPLE_BARCODE']),
                "test": list(self.response.iloc[test_index]['SAMPLE_BARCODE'])
            }
        np.save(CV_SPLIT_PATH, partition)
        return partition

    def get_config(self):
        """get data and its configurations for generators

        Returns:
            dict: "omics_dim": {"cnv": 123, "mutation": 12312 ...}
                  "drug_dim": {"rdkit2d": 200, "fingerprint": 881}
                  "omics_data": {'cnv': pd.DataFrame....}
                  "drug_data": {'rdkit2d': pd.DataFrame, 'fingerprint': pd.DataFrame}
                  "labels": {"ID": 1, "ID": 2, ...}
        """
        return {
            "omics_dim": {i:j.shape[1] for i,j in self.omics_data.items()},
            "drug_dim": {i:j.shape[1] for i,j in self.drug_info.drug_feature.items()},
            "omics_data": self.omics_data,
            "drug_data": self.drug_info.drug_feature,
            "labels": self.labels
        }

    def preprocess_experiment(self):
        print("Begin Preprocessing Experiment!")
        # Experiment Preprocess, align with celline_barcode and search pubchem_id
        print("Select Overlapping Cellines...")
        all_experiment = self.experiment[self.experiment['CELL_LINE_NAME'].isin(self.celline_barcode)]
        all_experiment.reset_index(drop=True, inplace = True)
        print(all_experiment.columns)
        print("Select Overlapping Cellines with available PubChem CIDs...")
        if (self.dataset == "GDSC"):
            all_experiment = all_experiment.join(self.drug_info.all_drugs, on="DRUG_NAME", how="inner")
            #all_experiment = all_experiment.join(self.drug_info.cid2smiles.set_index('CID'), on='CID', how='inner')
            all_experiment.reset_index(drop=True, inplace=True)
        elif (self.dataset == "CTRP"):
            all_experiment = all_experiment.join(self.drug_info.all_drugs, on="DRUG_NAME", how="inner")
            #all_experiment = all_experiment.join(self.drug_info.cid2smiles.set_index('CID'), on='CID', how='inner')
            all_experiment.reset_index(drop=True, inplace=True)

        print("Create Unique Sample Barcode...")
        sample_barcode = [f"{i[0]}_{i[1]}" for i in zip(all_experiment['CELL_LINE_NAME'], all_experiment['DRUG_NAME'])]
        all_experiment['SAMPLE_BARCODE'] = sample_barcode

        # Normalization if needed
        if self.target == "LN_IC50" or self.dataset == "CTRP":
            all_experiment[self.target]=(all_experiment[self.target]-all_experiment[self.target].min())/(all_experiment[self.target].max()-all_experiment[self.target].min())
        
        # Exclude outliers
        print("Exclude response value...")
        target = all_experiment[self.target].values
        df = pd.DataFrame(target)

        lower, median, upper = df.quantile([0.15,0.5,0.85]).values
        IQR = upper - lower
        lower_limit = lower - 1.5*IQR
        upper_limit = upper + 1.5*IQR

        all_experiment.loc[(all_experiment[self.target] < upper_limit.data) &
                        (all_experiment[self.target] > lower_limit.data)]

        all_experiment.reset_index(drop=True, inplace=True)

        # Update celline_barcode after exclusion
        self.celline_barcode = list(set(all_experiment['CELL_LINE_NAME']).intersection(self.celline_barcode))
        print("Experiment Done!")

        return all_experiment

    def preprocess_omics(self) -> dict:
        s = {}
        if 'gene_expression' in self.feature_contained:
            s['gene_expression'] = self.fpkm.loc[self.celline_barcode]
        if 'cnv' in self.feature_contained:
            s['cnv'] = self.cnv.loc[self.celline_barcode]
        if 'mutation' in self.feature_contained:
            s['mutation'] = self.mutation.loc[self.celline_barcode]
        if 'methylation' in self.feature_contained:
            s['methylation'] = self.methylation.loc[self.celline_barcode]
        return s

    def prepare_response(self, res) -> pd.DataFrame:
        response = pd.DataFrame()
        response['SAMPLE_BARCODE'] = self.processed_experiment['SAMPLE_BARCODE']
        response['RESPONSE'] = self.processed_experiment[res]
        response.reset_index(drop=True, inplace=True)
        return response

    def get_celline_barcode(self):
        slist = []
        s1 = set(self.experiment['CELL_LINE_NAME'])
        # slist.append(set(filtered_celline['Sample Name']) )
        if "gene_expression" in self.feature_contained:
            slist.append(set(self.fpkm.index))
        if "cnv" in self.feature_contained:
            slist.append(set(self.cnv.index))
        if "methylation" in self.feature_contained:
            slist.append(set(self.methylation.index))
        if "mutation" in self.feature_contained:
            slist.append(set(self.mutation.index))
        celline_barcode = list(s1.intersection(*slist))
        return celline_barcode
        
    def statistics_and_describe(self):
        print("Experiment: ")
        print(self.processed_experiment.shape)
        print(self.processed_experiment.columns)
        if "cnv" in self.feature_contained:
            print("Copy Number Variation: ")
            print(self.omics_data['cnv'].shape)
            print(self.omics_data['cnv'].columns)
        if "gene_expression" in self.feature_contained:
            print("Gene Expression FPKM: ")
            print(self.omics_data['gene_expression'].shape)
            print(self.omics_data['gene_expression'].columns)
        if "methylation" in self.feature_contained:
            print("Methylation: ")
            print(self.omics_data['methylation'].shape)
            print(self.omics_data['methylation'].columns)
        if "mutation" in self.feature_contained:
            print("Mutation: ")
            print(self.omics_data['mutation'].shape)
            print(self.omics_data['mutation'].columns) 

    def return_data(self) -> dict:
        return {
            "omics_data": self.omics_data,
            "response": self.response,
            "experiment": self.processed_experiment
        }
    
    def save(self):
        print("Save the dataset into hdf5 data format...")
        cid_list = list(set(i.split('_')[1] for i in self.response['SAMPLE_BARCODE']))
        celline_id = list(set(i.split('_')[0] for i in self.response['SAMPLE_BARCODE']))
        with h5py.File(HDF5_SAVE_PATH, 'w') as ds:
            if 'cnv' in self.feature_contained: 
                cnv=ds.create_group(name='cnv')
            if 'gene_expression' in self.feature_contained:
                gene_expression=ds.create_group(name='gene_expression')
            if 'mutation' in self.feature_contained:
                mutation=ds.create_group(name='mutation')
            if 'methylation' in self.feature_contained:
                methylation=ds.create_group(name='methylation')
            rdkit2d = ds.create_group(name='rdkit2d')
            fingerprint = ds.create_group(name='fingerprint')
            
            for i in celline_id:
                if 'cnv' in self.feature_contained:
                    cnv.create_dataset(name=i, data=self.omics_data['cnv'].loc[i].values, dtype=np.float32)
                if 'gene_expression' in self.feature_contained:
                    gene_expression.create_dataset(name=i, data=self.omics_data['gene_expression'].loc[i].values, dtype=np.float32)
                if 'mutation' in self.feature_contained:
                    mutation.create_dataset(name=i, data=self.omics_data['mutation'].loc[i].values, dtype=np.float32)
                if 'methylation' in self.feature_contained:
                    methylation.create_dataset(name=i, data=self.omics_data['methylation'].loc[i].values, dtype=np.float32)
            for j in cid_list:
                rdkit2d.create_dataset(name=str(j), data=self.drug_info.drug_feature['rdkit2d'].loc[j].values, dtype=np.float32)
                fingerprint.create_dataset(name=str(j), data=self.drug_info.drug_feature['fingerprint'].loc[j].values, dtype=np.float32)
            print("Done!")
        
class DataGenerator(keras.utils.Sequence):
    """DataGenerator, receive Dataset object and create generator used for model.fit

    Args:
        keras (keras.utils.Sequence): inherited
    """
    def __init__(self, 
                 sample_barcode, 
                 omics_data,
                 drug_data,
                 labels, 
                 omics_dim,
                 drug_dim,
                 batch_size=128, 
                 n_classes=2, 
                 shuffle=True,
                 ) -> None:
        super().__init__()
        self.sample_barcode = sample_barcode
        self.labels = labels
        self.batch_size = batch_size
        self.omics_dim = omics_dim
        self.drug_dim = drug_dim
        self.omics_data = omics_data
        self.drug_data = drug_data
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.sample_barcode) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sample_barcode))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sample_barcode_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        omics_feature = []
        X_rdkit2d = np.empty((self.batch_size, self.drug_dim['rdkit2d']), dtype=np.float32)
        X_fingerprint = np.empty((self.batch_size, self.drug_dim['fingerprint']), dtype=np.float32)
        omics_feature.append(X_fingerprint)
        omics_feature.append(X_rdkit2d)
        if 'cnv' in list(self.omics_data.keys()):
            X_cnv = np.empty((self.batch_size, self.omics_dim['cnv']), dtype=np.float32)
            omics_feature.append(X_cnv)
        if 'gene_expression' in list(self.omics_data.keys()):
            X_expr = np.empty((self.batch_size, self.omics_dim['gene_expression']), dtype=np.float32)
            omics_feature.append(X_expr)
        if 'mutation' in list(self.omics_data.keys()):
            X_mutation = np.empty((self.batch_size, self.omics_dim['mutation']), dtype=np.float32)
            omics_feature.append(X_mutation)
        if 'methylation' in list(self.omics_data.keys()): 
            X_methylation = np.empty((self.batch_size, self.omics_dim['methylation']), dtype=np.float32)
            omics_feature.append(X_methylation)
        
        y = np.empty((self.batch_size), dtype=int)

        # Generate data

        with h5py.File(HDF5_SAVE_PATH, mode='r') as ds:
        
            for i, ID in enumerate(sample_barcode_temp):
            # Store sample
                celline_id, cid = ID.split("_")
                if 'cnv' in list(self.omics_data.keys()): 
                    X_cnv[i,] = ds['cnv'][celline_id][()]
                if 'gene_expression' in list(self.omics_data.keys()):
                    X_expr[i,] = ds['gene_expression'][celline_id][()] 
                if 'mutation' in list(self.omics_data.keys()):
                    X_mutation[i,] = ds['mutation'][celline_id][()]
                if 'methylation' in list(self.omics_data.keys()): 
                    X_methylation[i,] = ds['methylation'][celline_id][()]
                X_rdkit2d[i,] = ds['rdkit2d'][cid][()]
                X_fingerprint[i,] = ds['fingerprint'][cid][()]
                y[i] = self.labels[ID]
        # print(X_cnv.shape, X_cnv.dtype, str(X_cnv))

        return omics_feature, y # keras.utils.to_categorical(y, num_classes=2)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        sample_barcode_temp = [self.sample_barcode[k] for k in indexes]

        # Generate data
        return self.__data_generation(sample_barcode_temp)
    
