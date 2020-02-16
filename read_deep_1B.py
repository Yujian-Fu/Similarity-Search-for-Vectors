import numpy as np
import os

index_range = 36
dimension = 96
fname =  '/home/y/yujianfu/similarity_search/datasets/Deep1B/'

def read_deep1B(scale_param):
    scale = 10**scale_param
    a = np.fromfile(os.path.join(fname, 'base_00'), dtype='int32')
    data_length = a.shape[0]
    number_of_file = int((dimension+1) * scale / data_length) + 1
    for j in (1, number_of_file):
        if j < 10:
            name = 'base_0' + str(j)
        b = np.fromfile(os.path.join(fname, name), dtype='int32')
        a = np.append(a, b)
    return a[0:(dimension+1)*scale].reshape(-1, dimension+1)[ : , 1:].copy().view('float32')

