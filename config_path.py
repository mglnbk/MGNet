import sys
import glob
from os.path import join, realpath

# data_path definition

RAW_FPKM_GDSC_PATH = realpath("data/raw_data/cellines_rnaseq_all_20220624/rnaseq_fpkm_20220624.csv")
RAW_CNV_GDSC_PATH = realpath("data/raw_data/celline_SNP6_cnv_gistics_20191101/cnv_abs_copy_number_picnic_20191101.csv")
RAW_EXPERIMENT_GDSC_PATH = realpath("data/raw_data/GDSC2_drug_dose_cellines_IC50s.xlsx")
RAW_METHYLATION_GDSC_PATH = realpath("data/raw_data/F2_METH_CELL_DATA.txt")
RAW_COMPOUND_GDSC_PATH = realpath("data/raw_data/all_compounds_screened.csv")
RAW_CELLINE_GDSC_PATH = realpath("data/raw_data/all_cellines_screened.xlsx")
RAW_PUBCHEMID_GDSC_PATH = realpath("data/raw_data/drug_info.csv")
RAW_SENTRIX2SAMPLE_GDSC_PATH = realpath("data/raw_data/methSampleId_2_cosmicIds.xlsx")

# model parameters settings
lr_rate = 1e-3
dropout_rate = .5

# choose from ['methylation', 'gene_expression', 'cnv', 'snv']
feature_contained = ['gene_expression', 'cnv', 'methylation']

# Choose from feature preprocessors ['SNF', 'Pretrained_beta-VAE', 'Pretrained_GCN', 'Transformer']
gene_feature_extractor = ['SNF', 'Transformer']

# choose from ['Pretrained_beta-VAE', 'Pretrained_GCN'] 
drug_feature_extractor = ['Pretrained_beta-VAE', 'Pretrained_GCN']

