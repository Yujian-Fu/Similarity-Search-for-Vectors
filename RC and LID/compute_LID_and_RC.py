
import sys
from fvecs_read import fvecs_read
import numpy as np
import scipy.io as sio
import random
import time
import math
import os

parameter_list = sys.argv[1:]
file_name = parameter_list[0]
instances_in_use = int(parameter_list[1])
dataset_name = parameter_list[2]
save_path = parameter_list[3]
file_type = parameter_list[4]
distance_length = 5000

print(parameter_list)
estimator_start = 10
estimator_end = 5000
estimator_gap = 50

if file_type == 'fvecs':
    data = fvecs_read(file_name)
elif file_type == 'npy':
    data = np.load(file_name)
elif file_type == 'mat':
    data = sio.load(file_name)
else:
    print('file type error, please choose one type from those three choices: fvecs, npy and mat ', file_type)


(instances, length) = data.shape

if length > instances:
    print('the size of this dataset should be transposed')

index = random.sample(list(range(0,instances)), instances_in_use)

distance_result = np.zeros((instances_in_use, distance_length))
relative_contrast = np.zeros((1, instances_in_use))

time_start = time.time()
count = 0

for i in index:
    vector = data[i, :]
    Distance = np.sqrt((vector**2).dot(np.ones((length, instances)))+np.ones(vector.size).dot((np.transpose(data)**2))-2*vector.dot(np.transpose(data)))
    Distance = np.sort(Distance)
    Distance = Distance[1:]
    distance_result[count, :] = Distance[ 0 :distance_length]
    Dmean = np.mean(Distance)
    Dmin = np.min(Distance)
    relative_contrast[0, count] = Dmean/Dmin
    count = count +1
    if np.mod(count, 100) == 0:
        print('now the count is', count, 'in', instances_in_use)

rc_mean = np.mean(relative_contrast)
rc_std = np.std(relative_contrast)
rc_meadian = np.std(relative_contrast)

np.save(os.path.join(save_path, 'distance matrix.npy'), distance_result)
np.save(os.path.join(save_path, 'relative_contrast.npy'), relative_contrast)

LID_MLE = np.zeros((1, instances_in_use))
LID_RV = np.zeros((1, instances_in_use))

k_estimators = np.arange(estimator_start, estimator_end, estimator_gap)

for k_estimator in k_estimators:
    for count in range(instances_in_use):
        Distance = distance_result[count, :]
        Distance = np.sort(Distance)
        IDx = 0
        #the MLE estimator
        for i in range(k_estimator):
            IDx = IDx + (1/k_estimator)*np.log(Distance[i]/Distance[k_estimator])
        IDx = -1/IDx
        LID_MLE[0, count] = IDx
        #the RV estimator
        numerator = np.log(k_estimator) - np.log(int(k_estimator/2))
        denominator = np.log(Distance[k_estimator]) - np.log(Distance[int(k_estimator/2)])
        IDrv = numerator/denominator
        LID_RV[0, count] = IDrv 

LID = np.array([LID_MLE, LID_RV])

np.save(os.path.join(save_path, 'LID.npy'), LID)





                                                  

    










