import pandas as pd
import numpy as np
from tensorflow import keras
import sys
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import *
from model.beta_vae import *
import subprocess
from model.utils import Drug


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
    print("Loading Methylation Data Done")
    return met_df.T

def load_data_snv(data_path) -> pd.DataFrame:
    print("Loading Mutations Data...")
    snv_df = pd.read_csv(data_path, low_memory=False)
    important_only = ['cds_disrupted','complex_sub','downstream', 'ess_splice','frameshift','missense','nonsense','silent','splice_region','start_lost','stop_lost','upstream']
    snv_df = snv_df[snv_df['effect'].isin(important_only)]
    df_table = pd.pivot_table(data=snv_df, 
                                index='model_name', 
                                columns='gene_symbol', 
                                values='effect',
                                aggfunc='count',
                                fill_value=0)
    print("Loading Mutations Data Done")
    return df_table

class Dataset:
    """DatasetClass

    Returns:
        _type_: _description_
    """
    # This class will facilitate the creation of Dataset
    def __init__(self, feature_contained = ['cnv', 'gene_expression'], dataset = 'GDSC'):
        self.feature_contained = feature_contained
        self.dataset = dataset

        # load multi-omics data
        if "cnv" in feature_contained:
            self.cnv = load_data_cna(RAW_CNV_GDSC_PATH)
        if "gene_expression" in feature_contained:
            self.fpkm = load_data_fpkm(RAW_FPKM_GDSC_PATH)
        if "methylation" in feature_contained:
            self.methylation = load_data_methylation(RAW_METHYLATION_GDSC_PATH)
        if "snv" in feature_contained:
            self.snv = load_data_snv(RAW_SNV_GDSC_PATH)

        self.drug_info = Drug()
        # self.celline = load_data_celline(RAW_CELLINE_GDSC_PATH, "GDSC")
        self.experiment = load_data_experiment(RAW_EXPERIMENT_GDSC_PATH, self.dataset)
        print(self.experiment.columns)
        # First to preprocess experiment data matrix !
        self.celline_barcode = self.get_celline_barcode()
        self.processed_experiment = self.preprocess_experiment()

        # Then preprocess omics data and response
        self.omics_data = self.preprocess_omics()
        self.response = self.prepare_response()

    def preprocess_experiment(self):
        print("Begin Preprocessing Experiment!")
        # Experiment Preprocess, align with celline_barcode and search pubchem_id
        print("Select Overlapping Cellines...")
        all_experiment = self.experiment[self.experiment['CELL_LINE_NAME'].isin(self.celline_barcode)]
        all_experiment.reset_index(drop=True, inplace = True)

        print("Select Overlapping Cellines with available PubChem CIDs...")
        if (self.dataset == "GDSC"):
            all_experiment = all_experiment.join(self.drug_info.gdsc_filter.set_index('SYNONYMS'), on='DRUG_NAME', how="inner")
            all_experiment = all_experiment.join(self.drug_info.cid2smiles.set_index('CID'), on='CID', how='inner')
            all_experiment.reset_index(drop=True, inplace=True)
        elif (self.dataset == "CTRP"):
            all_experiment = all_experiment.join(self.drug_info.ctrp_filter.set_index('SMILES'), on='SMILES', how="inner")
            all_experiment = all_experiment.join(self.drug_info.cid2smiles.set_index('CID'), on='CID', how='inner')
            all_experiment.reset_index(drop=True, inplace=True)

        print("Create Unique Sample Barcode...")
        sample_barcode = [f"{i[0]}_{i[1]}" for i in zip(all_experiment['CELL_LINE_NAME'], all_experiment['CID'])]
        all_experiment['SAMPLE_BARCODE'] = sample_barcode
        
        # exclude outliers
        print("Exclude response value...")
        ln_ic50 = all_experiment['AUC'].values
        df = pd.DataFrame(ln_ic50)

        lower, median, upper = df.quantile([0.15,0.5,0.85]).values
        IQR = upper - lower
        lower_limit = lower - 1.5*IQR
        upper_limit = upper + 1.5*IQR

        all_experiment.loc[(all_experiment['AUC'] < upper_limit.data) &
                        (all_experiment['AUC'] > lower_limit.data)]

        all_experiment.reset_index(drop=True, inplace=True)
        # 同时也要更新一下celline-barcode
        self.celline_barcode = list(set(all_experiment['CELL_LINE_NAME']).intersection(self.celline_barcode))
        print("Experiment Done!")

        return all_experiment

    def preprocess_omics(self) -> dict:
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
        response['AUC'] = self.processed_experiment['AUC']
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

    def return_data(self) -> dict:
        return {
            "omics_data": self.omics_data,
            "response": self.response,
            "experiment": self.processed_experiment
        }
    
    def return_feature(self, methods = ['SNF', 'beta-VAE']) -> dict:
        self.omics_data['methylation'].to_csv(PROCESSED_METHYLATION_GDSC_PATH)
        self.omics_data['gene_expression'].to_csv(PROCESSED_FPKM_GDSC_PATH)
        self.omics_data['cnv'].to_csv(PROCESSED_CNV_GDSC_PATH)
        self.omics_data['snv'].to_csv(PROCESSED_SNV_GDSC_PATH)

        # Omics_data Filter - SNF
        # if methods[0] == "SNF":
        #     print("Do Omics Integration!")
        #     subprocess.call([
        #         'Rscript', 
        #         R_SCRIPT_PATH,
        #         PROCESSED_CNV_GDSC_PATH, 
        #         PROCESSED_FPKM_GDSC_PATH, 
        #         PROCESSED_SNV_GDSC_PATH,
        #         PROCESSED_METHYLATION_GDSC_PATH]
        #         )
        
        similarity_df = pd.read_csv(SIM_PATH)
        similarity_df.drop(columns=['Unnamed: 0'], inplace=True)
        similarity_df.columns = self.omics_data['cnv'].index
        celline_feature = {}
        for i, celline in enumerate(similarity_df.columns):
            celline_feature[celline] = np.array(similarity_df.iloc[i].values)
        self.celline_feature = celline_feature
        # Drug Filter
        self.drug_feature_df = get_drug_feature()
        # Integration Filter
        s = []
        for i in self.processed_experiment['SAMPLE_BARCODE']:
            celline_name, pubchem_id = i.split('_')
            celline_feature_array = celline_feature[celline_name]
            drug_feature_array = self.drug_feature_df.loc[int(pubchem_id)].values
            combined_feature = np.hstack([celline_feature_array, drug_feature_array])
            s.append(combined_feature)
        return np.array(s)
