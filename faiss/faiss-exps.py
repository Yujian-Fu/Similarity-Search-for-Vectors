import faiss
import numpy as np
import random
from faiss_configuration import CONFIG
from fvecs_read import fvecs_read
from test_and_record import test_and_record


datasets = CONFIG.DATASET_PATH_LIST
queries = CONFIG.QUERY_PATH_LIST
train = CONFIG.TRAIN_PATH_LIST
assert len(datasets) == len(queries) == len(train)

num_datasets = len(datasets)

#compute SIFT dataset

for i in range(num_datasets):
    query_path = queries[i]
    dataset_path = datasets[i]
    train_path = train[i]
    dataset = fvecs_read(dataset_path)
    (instances, length) = dataset.shape
    query = fvecs_read(query_path)
    if train_path != ' ':
        train_dataset = fvecs_read(train_path)
    else:
        train_data = dataset[list(random.sample((range(instances), int(instances/10)))), :]
    dataset_name = query_path.split('/')[-1].split('.')[0]
    print(dataset.shape)
    test_and_record(dataset, query, train_dataset, dataset_name)

        
    

    
    


