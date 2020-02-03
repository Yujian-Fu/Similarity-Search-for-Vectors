import numpy as np 


def accumulate_distribution(accumulate_column, min_value, max_value, insert_value):
    instances = accumulate_column.shape[0]
    for i in range(instances):
        if min_value + (max_value - min_value)*i/instances > insert_value:
            accumulate_column[i, 0] += 1
            return accumulate_column
    print('error, insert value is even larger than the max')
    return accumulate_column


def compute_entropy(accumulate_column):
    instances = accumulate_column.shape[0]
    each_probability = 1 / (instances)
    entropy = 0

    for i in range(instances):
        if accumulate_column[i, 0] != 0:
            probability_i = accumulate_column[i, 0] * each_probability
            entropy += probability_i*np.log(probability_i)
    
    return -entropy

dataset_path_list = [
    '/home/y/yujianfu/similarity_search/datasets/Cifar/images_train.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_base.npy'
]

for dataset_path in dataset_path_list:
    dataset = np.load(dataset_path)
    instances, dimension = dataset.shape
    entropy = np.zeros((dimension, 1))
    for i in range(dimension):
        accumulate_column = np.zeros((instances, 1))
        feature_column = dataset[:, i]
        max_value = max(feature_column)
        min_value = min(feature_column)
        for j in range(instances):
            accumulate_column =  accumulate_distribution(accumulate_column, min_value, max_value, dataset[j, i])
        entropy[i, 0] = compute_entropy(accumulate_column)
        save_path = dataset_path.split('/')[0:-1]
        np.save(os.path.join(save_path, 'entropy.npy'), entropy)

