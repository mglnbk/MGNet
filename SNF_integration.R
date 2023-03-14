library(tidyverse)
library(naniar)
library(SNFtool)

args <- commandArgs(trailingOnly = TRUE)

cna_df <- read_csv(args[1])
fpkm_df <- read_csv(args[2])
snv_df <- read_csv(args[3])
methylation_df <- read_csv(args[4])


cna_df <- as.data.frame(cna_df[, -1])
fpkm_df <- as.data.frame(fpkm_df[, -1])
snv_df <- as.data.frame(snv_df[, -1])
methylation_df <- as.data.frame(methylation_df[, -1])

fpkm_df_normalized <- standardNormalization(fpkm_df)

cna_dist <- dist2(as.matrix(cna_df), as.matrix(cna_df))
fpkm_dist <- dist2(as.matrix(fpkm_df_normalized), as.matrix(fpkm_df_normalized))
snv_dist <- dist2(as.matrix(snv_df), as.matrix(snv_df))
methylation_dist <- dist2(as.matrix(methylation_df), as.matrix(methylation_df))

K <- 20 # number of neighbors, usually (10~30)
alpha <- 0.5 # hyperparameter, usually (0.3~0.8)
T <- 15 # Number of Iterations, usually (10~20)
W_cna <- affinityMatrix(cna_dist, K, alpha)
W_fpkm <- affinityMatrix(fpkm_dist, K, alpha)
W_snv <- affinityMatrix(snv_dist, K, alpha)
W_methylation <- affinityMatrix(methylation_dist, K, alpha)
W <- SNF(list(W_cna,W_fpkm), K, T)
str(W)
w_df <- as_tibble(W)
head(w_df)
write.csv(W, file = "data/processed_data/simlilarity_matrix.csv")
