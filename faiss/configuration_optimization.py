import numpy as np

# this is used to find the optimal setting for various algorithm implementations
# the evaluate dataset include ANN_SIFT10K, ANN_SIFT1M and SIFT10M, the dimension is 128 for all

search_set_list = CONFIG.DATASET_PATH_LIST
search_set_list = [search_set_list[2], search_set_list[1], search_set_list[7]]
query_set_list = CONFIG.QUERY_PATH_LIST
query_set_list = [query_set_list[2], query_set_list[1], query_set_list[7]]
learn_set_list = CONFIG.TRAIN_PATH_LIST
learn_set_list = [learn_set_list[2], learn_set_list[1], learn_set_list[7]]

# the tested algorithms include: IVFFlat, IVFPQ,  PQ, HNSWFlat, LSH
time_weight = 
recall_weight = 1 - time_weight

# parameter for IVFFlat: 
# the number of centroids in IVFFlat
nlist = [5, 10, 20 ,50, 100, 200, 400, 800, 1000]
# the number of centroids that will be visited in IVFFlat
nprobe = [1, 2, 3, 4, 5, 8, 10, 20, 50, 100, 200]


# parameters for HNSWFlat
#
num_of_neighbors = [4, 8, 12, 24, 36, 48, 64, 96]
#
efConstruction = [100, 200, 300, 400, 500, 600, 700, 800, 900]
#
efSearch = [10, 20, 40, 80, 120, 200, 400, 600, 800]



# parameters for IVFPQ:
#the number of 
nlist = 
# the number of 
code_size = 
#the number of 
nbits = 
#
nprobe = 


# parameters for PQ
#
M = 
#
nbits = 
#
nprobe = 




# parameters for LSH


