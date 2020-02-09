import numpy as np 
import os
import faiss

dataset_list = [


]

entropy_path = 

k_list = [1, 5, 10, 50]

for dataset_path in dataset_list:
    dataset = np.load(dataset_path)
    instances, dimension = dataset.shape
    dataset = np.transpose(dataset)

    entropy = np.load(entropy_path)
    performance = np.zeros(dimension, 2)
    index_brute = faiss.IndexFlatL2(instances)
    index_brute.add(dataset)
    for k in k_list:
        ground_truth = faiss.search
        


