# processes logs in the given directory and saves them in a subdirectory "inferred" 
# after running inference model on them

import os

inputPath = "/home/jkih/Music/sukwoo/Sphinx SVM_RBF c search 1.98 3 0.02/"
outputPath = inputPath + "inferred/"

# resistance to change
# e.g. 0.1 means changeover @ 0.4 and 0.6, or 0.5 +- 0.1
fuzzyness = 0.1
# decay value for EWMA
decay = 1.0/2

# read the raw output array
# process according to the logic values
# save in outputPath