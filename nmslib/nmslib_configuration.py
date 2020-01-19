from easydict import Easydict as edict
CONFIG = edict()

#dataset paths
CONFIG.DATASET_PATH_LIST = 
[


]


#parameters used in nmslib
CONFIG.K = range((10, 500), 10)
CONFIG.num_threads = 4
CONFIG.NUMBER_OF_EXPERIMENTS = 6
CONFIG.RECORDING_FILE = './recording_file/'

#index parameters, the most important onese
CONFIG.M = 15 
CONFIG.efC = 100
CONFIG.efS = 100