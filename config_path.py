from os.path import realpath
# data_path definition

# RAW_GDSC
RAW_EXPERIMENT_GDSC_PATH = realpath("data/raw_data/GDSC2_drug_dose_cellines_IC50s.xlsx")
RAW_COMPOUND_GDSC_PATH = realpath("data/raw_data/all_compounds_screened.csv")
RAW_CELLINE_GDSC_PATH = realpath("data/raw_data/all_cellines_screened.xlsx")

# Omics Path
RAW_FPKM_GDSC_PATH = realpath("data/raw_data/cellines_rnaseq_all_20220624/rnaseq_fpkm_20220624.csv")
RAW_CNV_GDSC_PATH = realpath("data/raw_data/celline_SNP6_cnv_gistics_20191101/cnv_abs_copy_number_picnic_20191101.csv")
RAW_METHYLATION_GDSC_PATH = realpath("data/raw_data/F2_METH_CELL_DATA.txt")
RAW_SENTRIX2SAMPLE_GDSC_PATH = realpath("data/raw_data/methSampleId_2_cosmicIds.xlsx")
RAW_SNV_GDSC_PATH = realpath("data/raw_data/mutations_all_20230202.csv")

### RAW_CTRP
RAW_EXPERIMENT_CTRP_PATH = realpath("data/raw_data/CTRPv2.0/v20.data.curves_post_qc.txt")
RAW_CELLINE_CTRP_PATH = realpath("data/raw_data/CTRPv2.0/v20.meta.per_cell_line.txt")
RAW_COMPOUND_CTRP_PATH = realpath("data/raw_data/CTRPv2.0/v20.meta.per_compound.txt")
RAW_META_EXPERIMENT_CTRP_PATH = realpath("data/raw_data/CTRPv2.0/v20.meta.per_experiment.txt")

## DRUG DATA PATH
RAW_DRUG_CTRP_PATH = realpath('data/processed_data/drugs/CTRP_SMILES2CIDS.txt')
RAW_DRUG_GDSC_PATH = realpath('data/processed_data/drugs/GDSC_SYNS2CIDS.txt')

## PROCESSED_GDSC_PATH
PROCESSED_FPKM_GDSC_PATH = realpath("data/processed_data/fpkm.csv")
PROCESSED_CNV_GDSC_PATH = realpath("data/processed_data/cnv.csv")
PROCESSED_METHYLATION_GDSC_PATH = realpath("data/processed_data/methylation.csv")
PROCESSED_SNV_GDSC_PATH = realpath("data/processed_data/mutation.csv")

## SCONDARY
R_SCRIPT_PATH = realpath("./SNF_integration.R")
SIM_PATH = realpath("data/processed_data/simlilarity_matrix.csv")
PUBCHEM_ID_SMILES_PATH = realpath("data/processed_data/drugs/pubchem_id-SMILES.csv")
PRETRAINED_BETA_VAE_PATH = realpath("utils/drug-molecule-generation-with-VAE")

## SAVE_PATH
HDF5_SAVE_PATH = realpath("data/processed_data/dataset.h5")
MODEL_PATH = realpath("./")


