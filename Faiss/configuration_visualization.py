import numpy as np   
import matplotlib.pyplot as plt
import heapq
from scipy import optimize
from scipy.interpolate import interp1d 
import glob
import os

def f_2(x, A, B, C):
    return A*x*x+B*x+C

def f_3(x, A, B, C, D):
    return A*x**3+B*x**2+C*x+D

def f_4(x, A, B, C, D, E):
    return A*x**4+B*x**3+C*x**2+D*x+E

def get_optimal(performance):
    assert performance.shape[1] == 2
    instances = performance.shape[0]

    index = []
    for i in range(instances):
        kick_out = 0
        for j in range(instances):
            if j != i and performance[j, 0] > performance[i, 0] - 0.01:
                if performance[j, 1] > performance [i, 1]:
                    kick_out = 1
        if kick_out == 0:
            index.append(i)
        
    return index

def draw_line_figure(optimal_setting, dataset, algo):
    #qps_new = spline(optimal_setting[:, 0], optimal_setting[:, 1], recall_new)
    
    recall = optimal_setting[:, 0]
    qps = optimal_setting[:, 1]
    '''
    A, B, C, D = optimize.curve_fit(f_3, recall, qps)[0]
    recall_new = np.zeros((recall.shape[0]*3-2, ))

    for i in range(recall.shape[0]-1):
        recall_new[3*i, ] = recall[i,]
        recall_new[3*i + 1, ] = 2*recall[i, ]/3 + recall[i+1,]/3
        recall_new[3*i + 2, ] = recall[i, ]/3 + 2*recall[i+1,]/3
    
    recall_new[-1, ] = recall[-1, ]
    
    print(recall, recall_new)
    qps_new = np.zeros((recall.shape[0]*3-2, ))

    for j in range(recall_new.shape[0]):
        qps_new[j,] = f_3(recall_new[j, ], A, B, C, D)
    '''
    if algo != 'HNSW':
        plt.plot(recall, qps, label = dataset+' ' + algo)
    else:
        plt.plot(recall, qps, label = dataset+' ' + algo)

    plt.legend(prop={'size':12})
    plt.grid(alpha = 0.5, linestyle= '--')
    plt.tight_layout()
    plt.yscale('log') 
    #plt.show()
    return recall, qps

def draw_scatter_figure(performance, dataset, algo):
    plt.figure()
    plt.scatter(performance[:, 0], performance[:, 1], label = dataset+' ' + algo, marker= '*')
    plt.xlabel('recall')
    plt.ylabel('qps') 
    plt.yscale('log') 
    plt.legend(prop= {'size':12})
    plt.grid(alpha = 0.5, linestyle = '--')
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(record_path, dataset, algo, 'configuration_scatter_500.png'))
    




dataset_list = [
    'ANN_SIFT10K',
    'ANN_SIFT1M',
    'ANN_GIST1M',
    'Deep10M',
    'SIFT10M'
    
    
]

algo_list = [
    'LSH',
    'HNSW',
    'IVFPQ',
    'annoy'
]

record_path = '/home/yujian/Downloads/exp_record_Feb16_1st/'


SIFT10K_list = [30.73, 5.2, 26.29, 25.49]
SIFT1M_list = [1994, ]
SIFT10M_list = [20818, 12566, 2723, 989]
Deep10M_list = [15757, 8748, 2438, 865]

i = -1
for algo in algo_list:
    cons_times = []
    x = np.arange(5)
    bar_width = 0.1

    '''
    plt.suptitle(dataset)
    k_list = [50,100,200,500,1000]
    axis1 = plt.subplot(1,3,1)
    axis2 = plt.subplot(1,3,2)
    axis3 = plt.subplot(1,3,3)
    axis1.set_title('recall')
    axis2.set_title('qps')
    axis3.set_title('dis_ratio')
    '''
    cons_time = []
    for dataset in dataset_list:
        #file_path = os.path.join(record_path, algo, dataset)
        #file_path = glob.glob(os.path.join(record_path, dataset, algo, '*.txt'))
        record_file = os.path.join(record_path, dataset, algo, 'record.txt')
        #if file_path[0].split('/')[-1] == 'optimal_configuration.txt':
            #file_path = file_path[1]
        #else:
            #file_path = file_path[0]
        
        #'/home/yujian/Downloads/similarity_search_datasets/exp_record/SIFT10K/LSH/qps_recall_LSH.txt'

        file = open(record_file, 'r')

        content = file.read()
        content = content.split('\n')[0:-1]
        instances = len(content)
        cons_time.append(float(content[0].split(' ')[2]))
    
    print(x, bar_width, cons_time)
    plt.bar(x+i*bar_width, cons_time, bar_width, label = algo)
    i += 1

plt.legend()
plt.xticks(x + bar_width / 4, dataset_list)
plt.yscale('log') 
plt.title('construction time')
plt.show()
'''
        recall = []
        qps = []
        dis_ratio = []

        for i in range(4,instances):
            each_line = content[i].split(' ')
            recall.append(float(each_line[-3]))
            qps.append(float(each_line[-1]))
            dis_ratio.append(float(each_line[-2]))
        
        
        axis1.plot(k_list, recall, label = algo)
        axis1.set_xlabel('K')
        axis2.plot(k_list, qps, label = algo)
        axis2.set_xlabel('K')
        axis3.plot(k_list, dis_ratio, label = algo)
        axis3.set_xlabel('K')


    plt.legend(loc = 'upper right')
    #plt.savefig(os.path.join(record_path, dataset,'performance.png'))
    plt.show()
    '''
        
        #draw_scatter_figure(performance, dataset, algo)

        #index = get_optimal(performance)


        #optimal_setting = performance[index, :]
'''
        new_file = open(os.path.join(record_path, dataset, algo,'optimal_configuration.txt'), 'w')
        for i in range(len(index)):
            new_file.write(content[index[i]]+'\n')

        optimal_setting = optimal_setting[optimal_setting[:, 0].argsort()]

        recall_new, qps_new = draw_line_figure(optimal_setting, dataset, algo)
        #plt.plot(optimal_setting[:, 0], optimal_setting[:, 1], label = dataset+ ' ' + algo)

        #fig = plt.figure()
        #plt.scatter(performance[:, 0], performance[:, 1], alpha = 0.8)
    plt.plot(1.0, 251, marker= '*', label = 'brute_force')
    plt.xlabel('recall')
    plt.ylabel('qps')
    plt.legend(prop={'size':12})
    plt.grid(alpha = 0.5, linestyle= '--')
    plt.tight_layout()
    plt.title('configuration skyline (k = 500)')
    plt.show()
    #plt.savefig(os.path.join(record_path, dataset, 'configuration_skyline.png'))
    '''













