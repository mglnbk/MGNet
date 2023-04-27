# MGNet
For undergraduate thesis design

1. 除去Outlier: Done
2. 加入甲基化数据: Done
3. 加入体细胞突变数据: Done
4. 药物数据CTRP: Done
5. 对训练集和数据集分别作Min-Max归一化: Done
6. 增添一组Validation实验: Done
7. 增添一个feature ablation实验, 对不同类型特征的重要性分析，可以做消融实验: Done
8. 增添一个对一种特定药物的重定位的研究实验: Done
9. Rank the drug tested for a specific drug：Done
10. 可以对网络中间特征进行聚类分析、热图分析后看看哪些细胞对那些药物敏感性更好，是否模型的中间特征具有预测效能：Done


Download URL Instruction
- GDSC2_drug_dose_cellines_IC50s.xlsx: https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC2_fitted_dose_response_24Jul22.xlsx
- all_cellines_screened.xlsx: https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/Cell_Lines_Details.xlsx
- all_compounds_screened.csv: https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/screened_compounds_rel_8.4.csv
- METH_CELL_DATA.txt.zip: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/methylation/METH_CELL_DATA.txt.zip
- methSampleId_2_cosmicIds.xlsx: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources///Data/otherAnnotations/methSampleId_2_cosmicIds.xlsx

From cell model passports
- mutations_all_20230202.csv: https://cog.sanger.ac.uk/cmp/download/mutations_all_20230202.zip
- model_list_2023_0306.csv: https://cog.sanger.ac.uk/cmp/download/model_list_20230306.csv
- cellines_rnaseq_all_20220624: https://cellmodelpassports.sanger.ac.uk/downloads
- celline_SNP6_cnv_gistics_20191101: https://cellmodelpassports.sanger.ac.uk/downloads

Note that the genomic data and compound data from CTRPv2 and GDSC are all processed from the same data source, CCLE(Cell Model Passport) and GDSC1000

As for the docs of single cell omics data, please read URL: https://depmap.sanger.ac.uk/documentation/datasets/

As for the data contained in data/processed_data/drugs/*, they are processed from the public interface provided by PubChem(https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi)

According to experiment result, 
For **GDSC**, lr=1e-3, threshold=.88, response=AUC, batch_size = 64 due to RAM limit
For **CTRP**, lr=1e-3, threshold=.58, response=AUC, batch_size = 64 due to RAM limit