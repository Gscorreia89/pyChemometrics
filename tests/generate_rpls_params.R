library("pls")


script.dir <- getSrcDirectory(function(x) {x})
setwd(script.dir)
# Use the python script to generate the datasets in case they don't exist
system("python gen_synthetic_datasets.py")
