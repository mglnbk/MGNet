library(tidyverse)
library(M3C)
library(ALL)
library(ConsensusClusterPlus)

# readin
cna_df <- read_csv("data_visual/cnv.csv")
expr_df <- read_csv("data_visual/gene_expression.csv")
mutation_df <- read_csv("data_visual/mutation.csv")
methylation_df <- read_csv("data_visual/methylation.csv")
encoded_df <- read_csv('ccl_encoded.csv')

# Different from python, col is sample, rows are features
expr_df <- t(expr_df[, c(2:ncol(expr_df))])
cna_df <- t(cna_df[, c(2:ncol(cna_df))])
mutation_df <- t(mutation_df[, c(2:ncol(mutation_df))])
methylation_df <- t(methylation_df[, c(2:ncol(methylation_df))])
encoded_df <- t(encoded_df[, c(2:ncol(encoded_df))])


mads <- apply(expr_df, 1, mad)
expr_df <- expr_df[rev(order(mads))[1:5000], ]
expr_df <- sweep(expr_df, 1, apply(expr_df, 1, median, na.rm = TRUE))

results <- ConsensusClusterPlus(expr_df, maxK = 10, reps = 50,
                                pItem = 0.8, pFeature = 1,
                                title = "title", clusterAlg = "hc",
                               distance = "pearson", seed = 1262118388.71279,
                               plot = "png")
results[[4]][["consensusTree"]]
heatmap(results[[1]][["consensusMatrix"]])
expr_df
