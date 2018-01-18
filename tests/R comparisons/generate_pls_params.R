# Not an automated test, just keeping this here for
# comparison with other algorithms

# Other libraries
library("pls")
library("ropls")
library("chemometrics")
library("pcaMethods")
library("mixOmics")

#script.dir <- getSrcDirectory(function(x) {x})
#setwd(script.dir)

# Use the python script to generate the datasets in case they don't exist
#system("python gen_synthetic_datasets.py")

# Load the two class discrimination dataset
pls_da_set <- read.csv("classification_twoclass.csv")

pls_da_set <- list(Class=pls_da_set$Class, X=as.matrix(pls_da_set[, 2:dim(pls_da_set)[2]]))

# fit the algorithm
pls_da <- plsr(Class ~ X, ncomp = 10, data = pls_da_set, validation = "LOO")