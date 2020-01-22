from easydict import EasyDict as edict
CONFIG = edict()

#directory for evaluation datasets
CONFIG.DATASET_PATH_LIST = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/gist_base.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/sift_base.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/siftsmall_base.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Cifar/images_train.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_base.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data.npy',
    '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature.npy'
]

#directory for evaluation queries
CONFIG.QUERY_PATH_LIST = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/gist_query.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/sift_query.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/siftsmall_query.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Cifar/images_train_query.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_query.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d_query.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data_query.npy',
    '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature_query.npy'
]

#path to evaluation train dataset
CONFIG.TRAIN_PATH_LIST = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/gist_learn.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/sift_learn.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/siftsmall_learn.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Cifar/images_train_learn.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_learn.fvecs',
    '/home/y/yujianfu/similarity_search/datasets/Glove/glove_840_300d_learn.npy',
    '/home/y/yujianfu/similarity_search/datasets/MNIST/MNIST_train_data_learn.npy',
    '/home/y/yujianfu/similarity_search/datasets/SIFT10M/SIFT10M_feature_learn.npy'
]

#parameters used in faiss
CONFIG.K = range(10, 500, 50) #the number of neighbors that you want to search
CONFIG.NLIST = 100 #the total number of cells
CONFIG.M = 4 #number of subquantilizers
CONFIG.NPROBE = 10 #number of cells to be visited
CONFIG.NUMBER_OF_EXPERIMENTS = 7 #number of index functions used in faiss
CONFIG.NBITS = 4 #how many bits that each sub-vector is encoded as
CONFIG.CODE_SIZE = 4 #used in PQ, the size of the code (how many bits)
CONFIG.NBITS = 8 #how many bits that each sub-vector is encoded as
CONFIG.CODE_SIZE = 8 #used in PQ, the size of the code (how many bits) can use 8, cannot use 4,(in IVFPQ) why???
CONFIG.NUM_OF_NEIGHBORS = 5 #used in HNSW, the number of neighbors of every point
CONFIG.DEPTH_CONSTRUCTION = 6 #the depth for construction and search
CONFIG.DEPTH_SEARCH = 6
CONFIG.FUNCTION_LIST = ['brute force', 'IVFFlat', 'IVFPQ', 'PQ', 'HNSWFlat', 'LSH', 'GPU']

#the name of recording file
CONFIG.RECORDING_FILE = './searching_record/'


