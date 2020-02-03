import numpy as np 


dataset_path_list = [


]

dataset = np.load(dataset_path)
instances, dimension = dataset.shape
for i in dimension:
     feature_column = dataset[:, i]
     max_value = 