1. 除去Outlier
2. 加入甲基化数据
3. 加入体细胞突变数据
4. 药物数据CTRP
5. 对训练集和数据集分别作Min-Max归一化
6. 增添一组Validation实验
7. 增添一个feature ablation实验
8. 增添一个对一种特定药物的重定位的研究实验
9. 对不同类型特征的重要性分析，可以做消融实验
10. 可以对药物进行聚类分析，聚类分析后看看该模型对那些药物敏感性更好
11. 最后除了分类还可以进行回归任务，再进行分析 regression versus classification


Download URL Instruction
- GDSC2_drug_dose_cellines_IC50s.xlsx: https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC2_fitted_dose_response_24Jul22.xlsx
- all_cellines_screened.xlsx: https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/Cell_Lines_Details.xlsx
- all_compounds_screened.csv: https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/screened_compounds_rel_8.4.csv
- SNP_6_cnv_gistics: https://cog.sanger.ac.uk/cmp/download/cnv_20191101.zip
- METH_CELL_DATA.txt.zip: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/methylation/METH_CELL_DATA.txt.zip
- methSampleId_2_cosmicIds.xlsx: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources///Data/otherAnnotations/methSampleId_2_cosmicIds.xlsx
- mutations_all_20230202.csv: https://cog.sanger.ac.uk/cmp/download/mutations_all_20230202.zip
- model_list_2023_0306.csv: https://cog.sanger.ac.uk/cmp/download/model_list_20230306.csv