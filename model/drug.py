import pandas as pd
import numpy as np
from os.path import realpath
import sys
from keras.models import load_model
import ast
from rdkit import Chem, RDLogger
try:
	from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except:
	raise ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus and pip install pandas-flavor")
from rdkit.Chem import BondType
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from utils.pybiomed import calcPubChemFingerAll
from config_path import *


RDLogger.DisableLog("rdApp.*")

SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]'

bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
bond_mapping.update(
    {0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC}
)
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

BATCH_SIZE = 100
EPOCHS = 10

VAE_LR = 5e-4
NUM_ATOMS = 120  # Maximum number of atoms

ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types
LATENT_DIM = 435  # Size of the latent space


def smiles_to_graph(smiles):
    # Converts SMILES to molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    features = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

    # loop over each atom in molecule
    # Ignore Pt and As Atom
    for atom in molecule.GetAtoms():
        if(atom.GetSymbol() == "Pt" or "As"):
            continue
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(ATOM_DIM)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return adjacency, features

def graph_to_molecule(graph):
    # Unpack graph
    adjacency, features = graph

    # RWMol is a molecule object intended to be edited
    molecule = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(features, axis=1) != ATOM_DIM - 1)
        & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
    )[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule

def smiles2rdkit2d(s):    
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = np.array(generator.process(s)[1:])
        NaNs = np.isnan(features)
        features[NaNs] = 0
    except:
        print('descriptastorus not found this smiles: ' + s + ' convert to all 0 features')
        features = np.zeros((200, ))
    return np.array(features)

def smiles2pubchem(s):
	try:
		features = calcPubChemFingerAll(s)
	except:
		print('pubchem fingerprint not working for smiles: ' + s + ' convert to 0 vectors')
		features = np.zeros((881, ))
	return np.array(features)

def get_drug_feature(method):
    df = pd.read_csv(PUBCHEM_ID_SMILES_PATH)
    df.drop_duplicates(subset=['CID'], inplace=True)
    if method == "beta-VAE":
        drug_AdjacencyTensor = []
        drug_FeatureTensor = []

        for i in df["CanonicalSMILES"]:
            _ad, _fe = smiles_to_graph(i)
            drug_AdjacencyTensor.append(_ad)
            drug_FeatureTensor.append(_fe)

        drug_AdjacencyTensor = np.array(drug_AdjacencyTensor)
        drug_FeatureTensor = np.array(drug_FeatureTensor)

        vae = load_model(PRETRAINED_BETA_VAE_PATH, compile=False)

        z_mean, _ = vae.encoder.predict([drug_AdjacencyTensor[:], drug_FeatureTensor[:]])
        drug_feature_df = pd.DataFrame(data = z_mean, index=df['CID'])

        return {'beta-VAE': drug_feature_df}
    
    elif method == "manual":
        fingerprint = []
        rdkit_2d_normalized = []
        for i in df['CanonicalSMILES']:
            fingerprint.append(smiles2pubchem(i))
            rdkit_2d_normalized.append(smiles2rdkit2d(i))
        fingerprint = np.array(fingerprint)
        rdkit_2d_normalized = np.array(rdkit_2d_normalized)
        drug_fingerprint_df = pd.DataFrame(data=fingerprint, index=df['CID'])
        drug_rdkit_2d_normalized_df = pd.DataFrame(data=rdkit_2d_normalized, index=df['CID'])

        return {
                "fingerprint": drug_fingerprint_df, 
                "rdkit2d": drug_rdkit_2d_normalized_df
                }

# CIDs are all extracted from PubChem Id Convertors
class Drug:
    def __init__(self, method='manual'):
        print("Begin loading drug data...")
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

        print(f"After filtering, GDSC has tested {len(self.gdsc_filter)} drugs \n CTRP has tested {len(self.ctrp_filter)} drugs")

        self.all_cids = pd.concat([df[['CID']], df2[['CID']]]).drop_duplicates(keep='first').reset_index(drop=True)

        print(f"After combining, unique cid number is {len(self.all_cids)}")
        import pubchempy as pcp
        te = pcp.get_properties(properties=['canonical_smiles'], 
                                identifier=list(self.all_cids['CID']),
                                namespace='cid')
        self.cid2smiles = pd.DataFrame(te).astype('str')
        self.cid2smiles.to_csv(PUBCHEM_ID_SMILES_PATH, index=None)

        self.drug_feature = get_drug_feature(method=method)

        print("Drug data loaded")