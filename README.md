# How to cluster datasets with important levels of noise or dropouts?

Let's start by defining the difference between noise and dropout:   

Dropout = dataset non 0 values appear as 0 (single cell RNA-seq data)  
Noise = the actual measured values have a certain additional noise (due to sensor calibration, experimental setup, etc)

This repository attempts to:
- explain the theoretical notions behing spectral clustering and self tuned spectral clustering
- implement the affinity matrix computation for self tuned spectral clustering 
- implement the eigenvalue gap heuristic for finding the optimal number of clusters
