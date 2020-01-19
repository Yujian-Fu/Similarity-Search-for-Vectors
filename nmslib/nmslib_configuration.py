from easydict import EasyDict as edict
CONFIG = edict()

#directory for evaluation datasets
CONFIG.DATASET_PATH_LIST = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/gist_base.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/sift_base.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/siftsmall_base.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Cifar/cifar-10-batches-py/images_train.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep_1M/deep1M_base.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data.npy',
    '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature.npy'
]

#directory for evaluation queries
CONFIG.QUERY_PATH_LIST = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/gist_query.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/sift_query.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/siftsmall_query.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Cifar/cifar-10-batches-py/images_train_query.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep_1M/deep1M_query.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d_query.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data_query.npy',
    '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature_query.npy'
]

#path to evaluation train dataset
CONFIG.TRAIN_PATH_LIST = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/gist_learn.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/sift_learn.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/siftsmall_learn.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Cifar/cifar-10-batches-py/images_train_learn.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep_1M/deep1M_learn.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d_learn.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data_learn.npy',
    '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature_learn.npy'
]



#parameters used in nmslib
CONFIG.K = range(10, 500, 10)
CONFIG.num_threads = 4
CONFIG.NUMBER_OF_EXPERIMENTS = 5
CONFIG.RECORDING_FILE = './recording_file/'

#index parameters, the most important onese
CONFIG.M = 15 
CONFIG.efC = 100
CONFIG.efS = 100