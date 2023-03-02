import sys
import glob
from os.path import join, realpath


raw_data_folder = "./data"
gdsc_celline_response_drug_IC50 = join(raw_data_folder, "GDSC2_drug_dose_cellines_IC50s")
processed_cnv = realpath("./processed_data/cnv.csv")
processed_fpkm = realpath("./processed_data/fpkm.csv")
