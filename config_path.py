from os.path import realpath
# data_path definition

## RAW_
RAW_FPKM_GDSC_PATH = realpath("data/raw_data/cellines_rnaseq_all_20220624/rnaseq_fpkm_20220624.csv")
RAW_CNV_GDSC_PATH = realpath("data/raw_data/celline_SNP6_cnv_gistics_20191101/cnv_abs_copy_number_picnic_20191101.csv")
RAW_EXPERIMENT_GDSC_PATH = realpath("data/raw_data/GDSC2_drug_dose_cellines_IC50s.xlsx")
RAW_METHYLATION_GDSC_PATH = realpath("data/raw_data/F2_METH_CELL_DATA.txt")
RAW_COMPOUND_GDSC_PATH = realpath("data/raw_data/all_compounds_screened.csv")
RAW_CELLINE_GDSC_PATH = realpath("data/raw_data/all_cellines_screened.xlsx")
RAW_PUBCHEMID_GDSC_PATH = realpath("data/raw_data/drug_info.csv")
RAW_SENTRIX2SAMPLE_GDSC_PATH = realpath("data/raw_data/methSampleId_2_cosmicIds.xlsx")
RAW_SNV_GDSC_PATH = realpath("data/raw_data/mutations_all_20230202.csv")

## DRUG DATA PATH
RAW_DRUG_CTRP_PATH = realpath('data/processed_data/drugs/CTRP_SMILES2CIDS.txt')
RAW_DRUG_GDSC_PATH = realpath('data/processed_data/drugs/GDSC_SYNS2CIDS.txt')

## PROCESSED_GDSC_PATH
PROCESSED_FPKM_GDSC_PATH = realpath("data/processed_data/fpkm.csv")
PROCESSED_CNV_GDSC_PATH = realpath("data/processed_data/cnv.csv")
PROCESSED_METHYLATION_GDSC_PATH = realpath("data/processed_data/methylation.csv")
PROCESSED_SNV_GDSC_PATH = realpath("data/processed_data/snv.csv")

## SCONDARY
R_SCRIPT_PATH = realpath("./SNF_integration.R")
SIM_PATH = realpath("data/processed_data/simlilarity_matrix.csv")
PUBCHEM_ID_SMILES_PATH = realpath("data/processed_data/drugs/pubchem_id-SMILES.csv")
PRETRAINED_BETA_VAE_PATH = realpath("utils/drug-molecule-generation-with-VAE")

# model parameters settings
lr_rate = 1e-3
dropout_rate = .5

# choose from ['methylation', 'gene_expression', 'cnv', 'snv']
feature_contained = ['gene_expression', 'cnv', 'methylation']

# Choose from feature preprocessors ['SNF', 'Pretrained_beta-VAE', 'Pretrained_GCN', 'Transformer']
gene_feature_extractor = ['SNF', 'Transformer']

# choose from ['Pretrained_beta-VAE', 'Pretrained_GCN'] 
drug_feature_extractor = ['Pretrained_beta-VAE', 'Pretrained_GCN']


if __name__ == '__main__':
    import pandas as pd

    print(pd.read_csv(PUBCHEM_ID_SMILES_PATH).columns)