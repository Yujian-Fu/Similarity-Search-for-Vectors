from easydict import EasyDict as edict
CONFIG = edict()

#directory for evaluation datasets
CONFIG.DATASET_PATH_LIST = [
    '/home/yujian/Downloads/similarity_search_datasets/siftsmall_base.fvecs',
    '/media/yujian/Seagate Backup Plus Drive/Datasets for Similarity Search/ANN_SIFT1M/sift/sift_base.fvecs',
    '/media/yujian/Seagate Backup Plus Drive/Datasets for Similarity Search/ANN_GIST1M/gist/gist_base.fvecs' ,

    '''
    'SIFT10M',
    'Deep1B',
    'MNIST',
    'Deep_feature',
    'SIFT_1B',
    '''
]

#directory for evaluation queries
CONFIG.QUERY_PATH_LIST = [
    '/home/yujian/Downloads/similarity_search_datasets/siftsmall_query.fvecs',
    '/media/yujian/Seagate Backup Plus Drive/Datasets for Similarity Search/ANN_SIFT1M/sift/sift_query.fvecs',
    '/media/yujian/Seagate Backup Plus Drive/Datasets for Similarity Search/ANN_GIST1M/gist/gist_query.fvecs',
    
    '''
    'SIFT10M',
    'Deep1B',
    'MNIST',
    'Deep_feature',
    'SIFT_1B',
    '''
]
CONFIG.TRAIN_PATH_LIST = [
    '/home/yujian/Downloads/similarity_search_datasets/siftsmall_learn.fvecs',
    '/media/yujian/Seagate Backup Plus Drive/Datasets for Similarity Search/ANN_SIFT1M/sift/sift_query.fvecs',
    '/media/yujian/Seagate Backup Plus Drive/Datasets for Similarity Search/ANN_GIST1M/gist/gist_query.fvecs',
    
    '''
    'SIFT10M',
    'Deep1B',
    'MNIST',
    'Deep_feature',
    'SIFT_1B',
    '''

]
#parameters used in faiss
CONFIG.K = 2 #the number of neighbors that you want to search
CONFIG.NLIST = 100 #the total number of cells
CONFIG.M = 8 #number of subquantilizers
CONFIG.NPROBE = 10 #number of cells to be visited
CONFIG.NUMBER_OF_EXPERIMENTS = 8 #number of index functions used in faiss
CONFIG.NBITS = 8 #how many bits that each sub-vector is encoded as

#the name of recording file
CONFIG.RECORDING_FILE = './searching_record'

#index functions in use
CONFIG.FUNCTION_LIST = ['']

