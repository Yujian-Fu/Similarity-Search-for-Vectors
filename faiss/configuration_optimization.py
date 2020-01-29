import numpy as np
from faiss-exps import read_dataset
import faiss

# this is used to find the optimal setting for various algorithm implementations
# the evaluate dataset include ANN_SIFT10K, ANN_SIFT1M and SIFT10M, the dimension is 128 for all

search_set_list = CONFIG.DATASET_PATH_LIST
search_set_list = [search_set_list[2], search_set_list[1], search_set_list[7]]
query_set_list = CONFIG.QUERY_PATH_LIST
query_set_list = [query_set_list[2], query_set_list[1], query_set_list[7]]
learn_set_list = CONFIG.TRAIN_PATH_LIST
learn_set_list = [learn_set_list[2], learn_set_list[1], learn_set_list[7]]

k = 100

# the tested algorithms include: IVFFlat, IVFPQ,  PQ, HNSWFlat, LSH
time_weight = 
recall_weight = 1 - time_weight 
# how to set a proper price function 
price_function = 

for i in range(len(search_set_list)):
    search_dataset = read_dataset(search_set_list[i])
    query_dataset = read_dataset(query_set_list[i])
    learn_dataset = read_dataset(learn_set_list[i])

    #dataset description 
    dimension = dataset.shape[1]
    query_length = query.shape[0]
    quantilizer = faiss.IndexFlatL2(dimension)

    #get groundtruth by brute force
    time_start = time.time()
    index = faiss.IndexFlatL2(dimension)
    index.add(search_dataset)
    dis_truth, ID_truth = index.search(query, k)
    time_end = time.time()
    time_brute = time_end - time_start


    # parameter for IVFFlat: 
    # the number of centroids in IVFFlat
    nlist_list = [5, 10, 20 ,50, 100, 200, 400, 800, 1000, 1200]
    # the number of centroids that will be visited in IVFFlat
    nprobe_list = [1, 3, 5, 8, 10, 20, 50, 80, 150, 200, 300]
    time_start = time.time()

    total_number = 0 
    for j in len(nlist):
        totoal_numer += np.sum(list(map(lambda x:x>=nlist_list[j], nprobe_list)))

    for nlist = 
    recall_record = np.zeros((total_number, 1))
    index = faiss.IndexIVFFlat(quantilizer, dimension, nlist) 
    index.probe = nprobe 
    assert not index.is_trained 
    index.train(learn_dataset) 
    assert index.is_trained 
    index.add(search_dataset) 
    search_result = index.search(query_dataset, k) 
    for j in query_length: 
        ground_truth = ID_truth[j, :] 
        search_result = search_result[j, :] 
        recall_record[j, 0] = len(set(ground_truth) & set(search_result))/len(set(ground_truth)) 
    
    recall = 0 
    for j in query_length:
        recall += recall_record[j, 1]
    recall = recall/query_length
    


    # parameters for HNSWFlat
    #
    num_of_neighbors = [4, 8, 12, 24, 36, 48, 64, 96]
    #
    efConstruction = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
    #
    efSearch = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]



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


