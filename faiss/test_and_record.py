import numpy as np
from faiss_configuration import CONFIG
import faiss
import time

function_list = ['brute force', 'IVFFlat', 'IVFPQ', 'PQ', 'HNSWFlat', 'LSH', 'GPU', 'GPUs']

def print_result(distance, ID, time_recorder, dataset_name):
    file = open(CONFIG.RECORDING_FILE + ' ' + dataset_name+'_function_time.txt', 'w')
    for i in range(CONFIG.NUMBER_OF_EXPERIMENTS):
        time_recorder[i, 0] = time_recorder[i, 1] - time_recorder[i, 0]
        file.write(function_list[i]+' '+str(time_recorder[i, 0])+'\n')

    final_record = np.concatenate((distance, ID), axis = 0)
    np.save(CONFIG.RECORDING_FILE + ' ' + dataset_name+'_dis_ID.npy', final_record)
    file.close()


def test_and_record(dataset, query, train_dataset, dataset_name):
    counter = 0
    time_recorder = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS, 2))
    dimension = dataset.shape[1]
    query_length = query.shape[0]
    quantilizer = faiss.IndexFlatL2(dimension)
    distance = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS,query_length, CONFIG.K))
    ID = np.zeros((CONFIG.NUMBER_OF_EXPERIMENTS,query_length, CONFIG.K))

    #search by brute force
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexFlatL2(dimension)
    index.add(dataset)
    D, I = index.search(query, CONFIG.K)
    print(dataset.shape, D.shape, I.shape, query.shape)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, CONFIG.K)
    time_recorder[counter, 1] = time.time()
    counter += 1


    #search by IVFFlat 
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexIVFFlat(quantilizer, dimension, CONFIG.NLIST)
    assert not index.is_trained
    index.train(train_dataset) #how much time does this train process take? can we ignore this?
    assert index.is_trained
    index.add(dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, CONFIG.K)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search by IVFPQ
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexIVFPQ(quantilizer, dimension, CONFIG.NLIST, CONFIG.M, CONFIG.NBITS)
    assert not index.is_trained
    index.train(train_dataset)
    assert index.is_trained
    index.add(dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, CONFIG.K)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search by PQ
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexPQ(dimension, CONFIG.M, CONFIG.NBITS)
    assert not index.is_trained
    index.train(train_dataset)
    assert index.is_trained
    index.add(dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, CONFIG.K)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search by HNSWFlat
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexHNSWFlat(dimension, CONFIG.M)
    index.add(train_dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, CONFIG.K)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search by LSH
    time_recorder[counter, 0] = time.time()
    index = faiss.IndexLSH(dimension, 2*dimension) 
        #why should we use 2*dimension? It says this is a improved LSH algorithm
    #assert not index.is_trained #why this shows the is_trained is true?
    index.train(train_dataset)
    assert index.is_trained
    index.add(dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, CONFIG.K)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search with GPU
    time_recorder[counter, 0] = time.time()
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, quantilizer)
    index_gpu.add(train_dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, CONFIG.K)
    time_recorder[counter, 1] = time.time()
    counter += 1

    #search with multiple GPUs
    time_recorder[counter, 0] = time.time()
    num_gpus = faiss.get_num_gpus()
    index = faiss.index_cpu_to_all_gpus(quantilizer)
    index.add(train_dataset)
    distance[counter,:,:], ID[counter,:,:] = index.search(query, CONFIG.K)
    time_recorder[counter, 1] = time.time()
    counter += 1

    assert counter == CONFIG.NUMBER_OF_EXPERIMENTS

    print_result(distance, ID, time_recorder, dataset_name)












    




