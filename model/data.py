import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from sklearn import preprocessing, model_selection
import sys
sys.path.append("/Users/sunzehui/Desktop/GraduationThesis/MGNet/") # Project root folder
from config_path import *

# 注：min-Max归一化需要在分割完训练集和测试集和Validation set之后再进行

def load_data_cna(data_path, data_source) -> pd.DataFrame:
    if data_source == "GDSC":
        all_cna = pd.read_csv(data_path, low_memory=False,
                      skiprows=lambda x: x in [0, 2])
        all_cna = all_cna.drop(columns=['model_name'])
        all_cna.rename(columns={"Unnamed: 1": "celline_barcode"}, inplace=True)
        all_cna.set_index('celline_barcode', inplace=True)
        all_cna = all_cna.T
        all_cna.fillna(0.0, inplace=True)
    return all_cna

def load_data_celline(celline_data_path, data_source) -> pd.DataFrame:
    if data_source == "GDSC":
        celline = pd.read_excel(celline_data_path, sheet_name=0)
    return celline

def load_data_fpkm(data_path, data_source) -> pd.DataFrame:
    if data_source == "GDSC":
        all_fpkm = pd.read_csv(data_path, low_memory=False,
                       skiprows=lambda x: x in [0, 2, 3, 4])
        all_fpkm = all_fpkm.drop(columns=['model_name'])
        all_fpkm.rename(columns={"Unnamed: 1": "celline_barcode"}, inplace=True)
        all_fpkm.set_index('celline_barcode', inplace=True)
        all_fpkm = all_fpkm.T
        all_fpkm.fillna(0.0, inplace=True)
    return all_fpkm

def load_data_experiment(data_path, data_source) -> pd.DataFrame:
    if data_source == "GDSC":
        df = pd.read_excel(data_path)
    return df

def load_data_pubchemid(data_path, data_source) -> pd.DataFrame:
    # 下面的过程是为了处理GDSC数据集中存在的一些药物不存在pubchem_id的情况
    # 将这些条目去除
    if data_source == "GDSC":
        drug_df = pd.read_csv(data_path, sep=',')
        import re
        nonnumber = re.compile(r'\D+')
        pubchem_id = list(set(drug_df['pubchem']))
        pubchem_id = [i.split(',')[0] if "," in i else i for i in pubchem_id]
        pubchem_id = [i for i in pubchem_id if re.findall(pattern=nonnumber, string=i).__len__()==0]
        drug_df = drug_df[drug_df['pubchem'].isin(pubchem_id)]
    return drug_df    

def load_data_methylation(data_path, data_source) -> pd.DataFrame:
    if data_source == "GDSC":
        met_df = pd.read_csv(data_path, sep = '\t', index_col=0)
        temp = pd.read_excel(RAW_SENTRIX2SAMPLE_GDSC_PATH, sheet_name=0)
        temp['Sentrix_Barcode'] = list("_".join([i,j]) for i, j in zip(temp['Sentrix_ID'].astype(str), temp['Sentrix_Position']))
        tb = dict(zip(temp['Sentrix_Barcode'], temp['Sample_Name']))
        met_df.rename(columns=tb, inplace=True)
    return met_df.T

def load_data_snv(data_path, data_source) -> pd.DataFrame:
    if data_source == "GDSC":
        snv_df = pd.read_csv(data_path, low_memory=False)
        important_only = ['cds_disrupted','complex_sub','downstream', 'ess_splice','frameshift','missense','nonsense','silent','splice_region','start_lost','stop_lost','upstream']
        snv_df = snv_df[snv_df['effect'].isin(important_only)]
        df_table = pd.pivot_table(data=snv_df, 
                                  index='model_name', 
                                  columns='gene_symbol', 
                                  values='effect',
                                  aggfunc='count',
                                  fill_value=0)
    return df_table

class Dataset:
    """DatasetClass

    Returns:
        _type_: _description_
    """
    # This class will facilitate the creation of Dataset
    def __init__(self, feature_contained = ['cnv', 'gene_expression']):
        self.feature_contained = feature_contained

        # load multi-omics data
        if "cnv" in feature_contained:
            self.cnv = load_data_cna(RAW_CNV_GDSC_PATH, "GDSC")
        if "gene_expression" in feature_contained:
            self.fpkm = load_data_fpkm(RAW_FPKM_GDSC_PATH, "GDSC")
        if "methylation" in feature_contained:
            self.methylation = load_data_methylation(RAW_METHYLATION_GDSC_PATH, "GDSC")
        if "snv" in feature_contained:
            self.snv = load_data_snv(RAW_SNV_GDSC_PATH, "GDSC")
        
        # load drug_df, celline and experiment
        self.drug_df = load_data_pubchemid(RAW_PUBCHEMID_GDSC_PATH, "GDSC")
        self.celline = load_data_celline(RAW_CELLINE_GDSC_PATH, "GDSC")
        self.experiment = load_data_experiment(RAW_EXPERIMENT_GDSC_PATH, "GDSC")
        
        # First to preprocess experiment data matrix !
        self.celline_barcode = self.get_celline_barcode()
        self.processed_experiment = self.preprocess_experiment()

        # Then preprocess omics data and response
        self.omics_data = self.preprocess_omics()
        self.response = self.prepare_response()

    def preprocess_experiment(self):
        # Experiment Preprocess, align with celline_barcode and search pubchem_id
        all_experiment = self.experiment.join(self.drug_df.set_index('drug_name'), on='DRUG_NAME', how="inner")
        all_experiment = all_experiment[all_experiment['CELL_LINE_NAME'].isin(self.celline_barcode)]
        all_experiment.reset_index(drop=True, inplace = True)

        # Add SMILES
        import pubchempy as pcp
        df = pcp.get_properties(properties=['canonical_smiles'], 
                                identifier=list(self.drug_df['pubchem']),
                                namespace='cid', )
        df = pd.DataFrame(df)
        df[['CID']]=df[['CID']].astype(str)
        lookup_table_cid_smiles = dict(zip(df['CID'], df['CanonicalSMILES']))
        all_experiment['SMILES']=[lookup_table_cid_smiles[i] for i in all_experiment['pubchem']]
        sample_barcode = [f"{i[0]}_{i[1]}" for i in zip(all_experiment['CELL_LINE_NAME'], all_experiment['pubchem'])]
        all_experiment['SAMPLE_BARCODE'] = sample_barcode
        
        # exclude outliers
        ln_ic50 = all_experiment['LN_IC50'].values
        df = pd.DataFrame(ln_ic50)

        lower, median, upper = df.quantile([0.15,0.5,0.85]).values
        IQR = upper - lower
        lower_limit = lower - 1.5*IQR
        upper_limit = upper + 1.5*IQR

        all_experiment.loc[(all_experiment['LN_IC50'] < upper_limit.data) &
                        (all_experiment['LN_IC50'] > lower_limit.data)]

        all_experiment.reset_index(drop=True, inplace=True)
        # 同时也要更新一下celline-barcode以防万一
        self.celline_barcode = list(set(all_experiment['CELL_LINE_NAME']).intersection(self.celline_barcode))


        return all_experiment

    def preprocess_omics(self):
        s = {}
        if 'gene_expression' in self.feature_contained:
            s['gene_expression'] = self.fpkm.loc[self.celline_barcode]
        if 'cnv' in self.feature_contained:
            s['cnv'] = self.cnv.loc[self.celline_barcode]
        if 'snv' in self.feature_contained:
            s['snv'] = self.snv.loc[self.celline_barcode]
        if 'methylation' in self.feature_contained:
            s['methylation'] = self.methylation.loc[self.celline_barcode]
        return s

    def prepare_response(self) -> pd.DataFrame:
        response = pd.DataFrame()
        response['sample_barcode'] = self.processed_experiment['SAMPLE_BARCODE']
        response['LN_IC50'] = self.processed_experiment['LN_IC50']
        response.reset_index(drop=True, inplace=True)
        return response

    def get_celline_barcode(self):
        # 寻找有数据的celline
        filtered_celline = self.celline.loc[
            (self.celline['Whole Exome Sequencing (WES)'] == "Y") &
            (self.celline['Methylation'] == "Y") &
            (self.celline['Gene Expression'] == "Y") &
            (self.celline['Copy Number Alterations (CNA)'] == "Y") &
            (self.celline['Drug\nResponse'] == "Y")
        ]
        slist = []
        s1 = set(self.experiment['CELL_LINE_NAME'])
        slist.append(set(filtered_celline['Sample Name']) )
        if "gene_expression" in self.feature_contained:
            slist.append(set(self.fpkm.index))
        if "cnv" in self.feature_contained:
            slist.append(set(self.cnv.index))
        if "methylation" in self.feature_contained:
            slist.append(set(self.methylation.index))
        if "snv" in self.feature_contained:
            slist.append(set(self.snv.index))
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
        if "snv" in self.feature_contained:
            print("Mutation: ")
            print(self.omics_data['snv'].shape)
            print(self.omics_data['snv'].columns) 

    def return_raw_data(self, scale = True) -> dict:
        return 

    def return_data(self) -> dict:
        return {
            "omics_data": self.omics_data,
            "response": self.response,
            "experiment": self.processed_experiment
        }
    
if __name__ == "__main__":
    d = Dataset()
    d.statistics_and_describe()
    s=d.return_data()
    print(s['experiment'].columns)