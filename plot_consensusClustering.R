library(tidyverse)
library(M3C)
library(ALL)
library(ConsensusClusterPlus)

# readin
encoded_df <- read_csv('result/ccl_encoded.csv')
drug_encoded_df <- read_csv('result/drug_encoded.csv')
mofa_df <- read_csv('data/processed_data/simlilarity_matrix.csv')

# Different from python, col is sample, rows are features
mofa_df <- mofa_df[, c(2:ncol(mofa_df))]
colnames(mofa_df) <- colnames(encoded_df)
mofa_df <- as.matrix(mofa_df)
encoded_df <- as.matrix(encoded_df)
drug_encoded_df <- as.matrix(drug_encoded_df)
encoded_df

drug_results <- ConsensusClusterPlus(drug_encoded_df, maxK = 10, reps = 50,
                                pItem = 0.8, pFeature = 1,
                                title = "DRUG", clusterAlg = "hc",
                               distance = "pearson", seed = 1262118388.71279,
                               plot = "png")
encoded_results <- ConsensusClusterPlus(encoded_df, maxK = 10, reps = 50,
                                        pItem = 0.8, pFeature = 1,
                                        title = "ENCODED", clusterAlg = "hc",
                                        distance = "pearson", seed = 1262118388.71279,
                                        plot = "png")
mofa_results <- ConsensusClusterPlus(mofa_df, maxK = 10, reps = 50,
                                        pItem = 0.8, pFeature = 1,
                                        title = "MOFA", clusterAlg = "hc",
                                        distance = "pearson", seed = 1262118388.71279,
                                        plot = "png")

mofa_clrs = as.tibble(mofa_results[[4]]$consensusClass)
mofa_clrs = t(mofa_clrs)
colnames(mofa_clrs) <- colnames(mofa_df)
mofa_clrs <- as_tibble(mofa_clrs)

encoded_clrs = as.tibble(encoded_results[[4]]$consensusClass)
encoded_clrs = t(encoded_clrs)
colnames(encoded_clrs) <- colnames(encoded_df)
encoded_clrs <- as_tibble(encoded_clrs)

drug_clrs = as.tibble(drug_results[[5]]$consensusClass)
drug_encoded_clrs = t(drug_clrs)
colnames(drug_encoded_clrs) <- colnames(drug_encoded_df)
drug_clrs <- as_tibble(drug_encoded_clrs)

write_csv(mofa_clrs ,file='./mofa_clrs.csv')
write_csv(encoded_clrs ,file='./encoded_clrs.csv')
write_csv(drug_clrs, file='./drug_clrs.csv')
