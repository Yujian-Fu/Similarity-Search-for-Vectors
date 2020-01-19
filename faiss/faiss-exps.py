import faiss
import numpy as np
from faiss_configuration import CONFIG
from fvecs_read import fvecs_read
from test_and_record import test_and_record

def read_dataset(file_name):
    if file_name.split('.')[-1] == 'npy':
        file = np.load(file_name)
    elif file_name.split('.')[-1] == 'fvecs':
        file = fvecs_read(file_name)
    else:
        print ('the file name', file_name, 'is wrong!')

    return file


datasets = CONFIG.DATASET_PATH_LIST
queries = CONFIG.QUERY_PATH_LIST
train = CONFIG.TRAIN_PATH_LIST
assert len(datasets) == len(queries) == len(train)

num_datasets = len(datasets)


for i in range(1:num_datasets):
    query_path = queries[i]
    dataset_path = datasets[i]
    train_path = train[i]

    dataset = read_dataset(dataset_path)
    query_set = read_dataset(query_path)
    train_set = read_dataset(train_path)

    (instances, length) = dataset.shape

    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    print(dataset.shape, dataset_name)

    for k in CONFIG.K:
        print('now computing K = ', k)
        test_and_record(dataset, query_set, train_set, dataset_name, k)

        
    

    
    


