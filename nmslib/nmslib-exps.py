import nmslib
import numpy as np  
from nmslib_configuration import CONFIG
from test_and_record import test_and_record


def fvecs_read(filename):
    ffile = np.fromfile(filename, dtype = np.float32)
    if ffile.size == 0:
        return zeros((0, 0))
    dimension = ffile.view(np.int32)[0]
    assert dimension > 0
    ffile = ffile.reshape(-1, 1+dimension)
    if not all(ffile.view(np.int32)[:, 0] == dimension):
        raise IOError('Non-uniform vector sizes in ' + filename)
    ffile = ffile[:, 1:]
    ffile = ffile.copy()
    return ffile

def read_dataset(filename):
    if filename.split('.')[-1] == 'npy':
        file = np.load(filename)
    elif filename.split('.')[-1] == 'fvecs':
        file = fvecs_read(filename)
    else:
        print('the file name', file_name, 'is wrong!')

    return file


datasets = CONFIG.DATASET_PATH_LIST
queries = CONFIG.QUERY_PATH_LIST
train = CONFIG.TRAIN_PATH_LIST

assert len(datasets) == len(queries) == len(train)

num_datasets = len(datasets)



for i in range(num_datasets):
    query_path = queries[i]
    dataset_path = datasets[i]
    train_path = train[i]

    dataset = read_dataset(dataset_path)
    query_set = read_dataset(query_path)
    train_path = read_dataset(train_path)

    (instances, length) = dataset.shape

    dataset_name = query_path.split('/')[-1].split('.')[0]
    if instances < length:
        print('the number of instances is smaller than the dimension, is it wrong?')
    print(dataset.shape, dataset_name)

    for k in CONFIG.K:
        print('now computing k = ', k)
        test_and_record(dataset, query_set, train_set, dataset_name, k)
