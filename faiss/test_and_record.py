import numpy as np
from faiss_configuration import CONFIG
import faiss
import time
import os


def print_result(distance, ID, time_recorder, dataset_name, k):
    path = os.path.join(CONFIG.RECORDING_FILE, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(CONFIG.RECORDING_FILE, dataset_name, str(k))
    if not os.path.exists(path):
        os.makedirs(path)

    file = open(os.path.join(path, dataset_name+'_function_time.txt'), 'w')
    assert len(CONFIG.FUNCTION_LIST) == CONFIG.NUMBER_OF_EXPERIMENTS
    file.write('the number of query: '+str(distance.shape[1]) + '\n')
    for i in range(CONFIG.NUMBER_OF_EXPERIMENTS):
        time_recorder[i, 0] = time_recorder[i, 1] - time_recorder[i, 0]
        file.write(CONFIG.FUNCTION_LIST[i]+' '+str(time_recorder[i, 0])+'\n')

    #np.save(os.path.join(path, dataset_name+'_dis.npy'), distance)
    np.save(os.path.join(path, dataset_name+'_ID.npy'), ID)
    file.close()


def test_and_record(dataset, query, train_dataset, dataset_name, k):
    counter = 0
    time_recorder = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS, 2))
    dimension = dataset.shape[1]
    query_length = query.shape[0]
    quantilizer = faiss.IndexFlatL2(dimension)
    distance = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS, query_length, k))
    ID = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS,query_length, k))

    #search by brute force
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexFlatL2(dimension)
    index.add(dataset)
    D, I = index.search(query, k)
    print(dataset.shape, D.shape, I.shape, query.shape)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, k)
    time_recorder[counter, 1] = time.time()
    counter += 1


    #search by IVFFlat 
    #built the inverse index and search some of the cells
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexIVFFlat(quantilizer, dimension, CONFIG.NLIST)
    index.probe = CONFIG.NPROBE #set this parameter by handcraft, or it will be set to 1 by default
    assert not index.is_trained
    index.train(train_dataset) #how much time does this train process take? can we ignore this?
    assert index.is_trained
    index.add(dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, k)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search by IVFPQ
    time_recorder[counter, 0] = time.time()
    #not sure what is the NBITS here? any difference to CODE_SIZE?
    #dimension should be a multiple of NBITS 
    index = faiss.IndexIVFPQ(quantilizer, dimension, CONFIG.NLIST, CONFIG.CODE_SIZE, CONFIG.NBITS)
    index.nprobe = CONFIG.NPROBE # the number of cells visited
    assert not index.is_trained
    index.train(train_dataset)
    assert index.is_trained
    index.add(dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, k)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search by PQ
    time_recorder[counter, 0] = time.time()
    #should NBITS the same as IVFPQ?
    index = faiss.IndexPQ(dimension, CONFIG.M, CONFIG.NBITS)
    index.probe = CONFIG>NPROBE
    assert not index.is_trained
    index.train(train_dataset)
    assert index.is_trained
    index.add(dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, k)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search by HNSWFlat
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexHNSWFlat(dimension, CONFIG.NUM_OF_NEIGHBORS)
    index.efConstruction = CONFIG.DEPTH_CONSTRUCTION
    index.efSearch = CONFIG.DEPTH_SEARCH
    index.add(train_dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, k)
    time_recorder[counter, 1] = time.time()
    counter += 1


    #search by LSH
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexLSH(dimension, 2*dimension) 
        #why should we use 2*dimension? It says this is a improved LSH algorithm
    #assert not index.is_trained #why this shows the is_trained is true? even not trained, you can add dataset to the index
    index.train(train_dataset)
    assert index.is_trained
    index.add(dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, k)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search with GPU
    time_recorder[counter, 0] = time.time()
    #why the GPU based search is not brute force search? the recall should be 1
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, quantilizer)
    index_gpu.add(train_dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, k)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search with multiple GPUs
    '''
    time_recorder[counter, 0] = time.time()
    num_gpus = faiss.get_num_gpus()
    index = faiss.index_cpu_to_all_gpus(quantilizer)
    index.add(train_dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, k)
    time_recorder[counter, 1] = time.time()
    counter += 1
    '''

    assert counter == CONFIG.NUMBER_OF_EXPERIMENTS

    print_result(distance, ID, time_recorder, dataset_name, k)












    




