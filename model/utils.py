import pandas as pd
import sys
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from config_path import RAW_DRUG_GDSC_PATH, RAW_DRUG_CTRP_PATH
from config_path import PUBCHEM_ID_SMILES_PATH
from sklearn.metrics import pairwise_distances
import numpy as np

# CIDs are all extracted from PubChem Id Convertors
class Drug:
    def __init__(self) -> None:
        
        df = pd.read_csv(RAW_DRUG_CTRP_PATH, 
                         sep='\t', 
                         dtype=('str', 'str')
                         )
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df2 = pd.read_csv(RAW_DRUG_GDSC_PATH, 
                          sep='\t', 
                          dtype=('str', 'str')
                          )
        df2.drop_duplicates(subset=['SYNONYMS'], keep='first', inplace=True)
        df2.dropna(inplace=True)
        df2.reset_index(drop=True, inplace=True)

        self.gdsc_filter = df2 # cols = ['SYNONYMS', 'CID']
        self.ctrp_filter = df  # cols = ['SMILES', 'CID']

        self.all_cids = pd.concat([df[['CID']], df2[['CID']]]).drop_duplicates(keep='first').reset_index(drop=True)

        import pubchempy as pcp
        te = pcp.get_properties(properties=['canonical_smiles'], 
                                identifier=list(self.all_cids['CID']),
                                namespace='cid')
        self.cid2smiles = pd.DataFrame(te).astype('str')
        self.cid2smiles.to_csv(PUBCHEM_ID_SMILES_PATH, index=None)

def sim_matrix(df):
    dist_matrix = pairwise_distances(df.values, metric='euclidean')
    gamma = 1/df.values.shape[1]
    gamma = 0.001
    return np.exp(-np.square(dist_matrix) * gamma)
