import numpy as np 
import os 
import faiss
import time    

dataset_list = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_base.npy', 
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_base.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_base.npy'
]

query_list = [
    [
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_SIFT10K/LID_MLE_500/large_LID.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_SIFT10K/LID_MLE_500/small_LID.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_SIFT10K/RC/large_RC.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_SIFT10K/RC/small_RC.npy'
    ]
    ,
    [
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_SIFT1M/LID_MLE_500/large_LID.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_SIFT1M/LID_MLE_500/small_LID.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_SIFT1M/RC/large_RC.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_SIFT1M/RC/small_RC.npy'
    ]
    ,
    [
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_GIST1M/LID_MLE_500/large_LID.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_GIST1M/LID_MLE_500/small_LID.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_GIST1M/RC/large_RC.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/ANN_GIST1M/RC/small_RC.npy'
    ]
    ,
    [
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/deep1M/LID_MLE_500/large_LID.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/deep1M/LID_MLE_500/small_LID.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/deep1M/RC/large_RC.npy',
        '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset/deep1M/RC/small_RC.npy'
    ]
]

learn_list = [
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT10K/SIFT10K_train.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_SIFT1M/SIFT1M_train.npy',
    '/home/y/yujianfu/similarity_search/datasets/ANN_GIST1M/GIST1M_learn.npy',
    '/home/y/yujianfu/similarity_search/datasets/deep1M/deep1M_learn.fvecs'
]

save_path = '/home/y/yujianfu/similarity_search/datasets/Selected_Dataset_performance/'

query_k_list = [5, 10, 20, 40, 80, 120, 200, 400, 600, 800, 1000]

IVFFlat_list = [[50, 8]]     
# SIFT10K 10, 5 10,3 20,8 5,3 5,1   SIFT1M 50,10  50,5 50,8 20, 3 20, 5   GIST1M 50, 5  50, 8  100, 10  50, 10  100, 8  
# deep1M 100, 10 100, 8 50, 8 50, 10 50, 5  SIFT10M 100,10 50, 8 200, 8 200, 20 200, 5
IVFPQ_list = [[100, 64, 50]]
# deep1M 400,64, 50  100,64,20  100, 64, 50  200,64,20 200,64,50  GIST1M 100,64,50 200,64,150 10,64,8 20,64,8 50,64,10
#  
HNSW_list = [[16, 100, 20]]
LSH_List = [[8192]]
PQ_list = [[128, 8]]


for i in range(len(dataset_list)):

    dataset_name = dataset_list[i].split('/')[-1].split('_')[0]

    search_dataset = np.load(dataset_list[i])
    learn_dataset = np.load(learn_list[i])

    if not os.path.exists(os.path.join(save_path, dataset_name, 'IVFFlat')):
        os.makedirs(os.path.join(save_path, dataset_name, 'IVFFlat'))
    IVFFlat_file = open(os.path.join(save_path, dataset_name, 'IVFFlat', 'IVFFlat.txt'), 'w')

    if not os.path.exists(os.path.join(save_path, dataset_name, 'IVFPQ')):
        os.makedirs(os.path.join(save_path, dataset_name, 'IVFPQ'))
    IVFPQ_file = open(os.path.join(save_path, dataset_name, 'IVFPQ', 'IVFPQ.txt'), 'w')

    if not os.path.exists(os.path.join(save_path, dataset_name, 'HNSW')):
        os.makedirs(os.path.join(save_path, dataset_name, 'HNSW'))
    HNSW_file = open(os.path.join(save_path, dataset_name, 'HNSW', 'HNSW.txt'), 'w')

    if not os.path.exists(os.path.join(save_path, dataset_name, 'LSH')):
        os.makedirs(os.path.join(save_path, dataset_name, 'LSH'))
    LSH_file = open(os.path.join(save_path, dataset_name, 'LSH', 'LSH.txt'), 'w')

    if not os.path.exists(os.path.join(save_path, dataset_name, 'PQ')):
        os.makedirs(os.path.join(save_path, dataset_name, 'PQ'))
    PQ_file = open(os.path.join(save_path, dataset_name, 'PQ', 'PQ.txt'), 'w')

    dimension = search_dataset.shape[1]
    time_1 = time.time()
    index = faiss.IndexFlatL2(dimension)
    index.add(search_dataset)
    time_2 = time.time()
    time_brute_con = time_2 - time_1


    #the build phase
    for j in range(len(IVFFlat_list)):
        #build for IVFflat
        param_IVFFlat = IVFFlat_list[j]
        time_1 = time.time()
        quantilizer = faiss.IndexFlatL2(dimension)
        index_IVF = faiss.IndexIVFFlat(quantilizer, dimension, param_IVFFlat[0])
        index_IVF.probe = param_IVFFlat[1]
        assert not index.is_trained
        index_IVF.train(learn_dataset)
        assert index.is_trained 
        index_IVF.add(search_dataset)
        time_2 = time.time()
        time_IVF_con = time_2 - time_1
        IVFFlat_file.write('time_con '+str(param_IVFFlat[0])+' '+str(param_IVFFlat[1])+' '+str(time_IVF_con)+'\n')
        print('finish build IVFFlat')

        #build for IVFPQ
        param_IVFPQ = IVFPQ_list[j]
        time_1 = time.time()
        quantilizer = faiss.IndexFlatL2(dimension)
        index_IVFPQ = faiss.IndexIVFPQ(quantilizer, dimension, param_IVFPQ[0], param_IVFPQ[1], 8)
        index_IVFPQ.nprobe = param_IVFPQ[2]
        assert not index_IVFPQ.is_trained 
        index_IVFPQ.train(learn_dataset)
        assert index_IVFPQ.is_trained
        index_IVFPQ.add(search_dataset)
        time_2 = time.time()
        time_IVFPQ_con = time_2 - time_1
        IVFPQ_file.write('time_con nlist '+ str(param_IVFPQ[0]) + ' code_size ' + str(param_IVFPQ[1]) + ' nprobe ' + str(param_IVFPQ[2]) + ' ' + str(time_IVFPQ_con) + '\n')
        print('finish build IVFPQ')

        #build for HNSW
        param_HNSW = HNSW_list[j]
        time_1 = time.time()
        index_HNSW = faiss.IndexHNSW(dimension, param_HNSW[0])
        index_HNSW.hnsw.efConstruction = param_HNSW[1]
        index_HNSW.hnsw.efSearch = param_HNSW[2]
        index_HNSW.add(search_dataset)
        time_2 = time.time()
        time_HNSW_con = time_2 - time_1
        HNSW_file.write('time_con n_neighbor '+ str(param_HNSW[0]) + ' efConstruction ' + str(param_HNSW[1]) + ' efSearch ' + str(param_HNSW[2]) + ' ' + str(time_HNSW_con) + '\n')
        print('finish build HNSW')

        #build for LSH
        param_LSH = LSH_List[j]
        time_1 = time.time()
        index_LSH = faiss.IndexLSH(dimension, param_LSH[0])
        index_LSH.train(learn_dataset)
        index_LSH.add(search_dataset)
        time_2 = time.time()
        time_LSH_con = time_2 - time_1
        LSH_file.write('time_con nbits '+str(param_LSH[0]) +' '+str(time_LSH_con)+'\n')
        print('finish build LSH')

        #parameter for PQ
        param_PQ = PQ_list[j]
        time_1 = time.time()
        index_PQ = faiss.IndexPQ(dimension, param_PQ[0], param_PQ[1])
        assert not inex_PQ.is_trained 
        index_PQ.learn(learn_dataset)
        assert index_PQ.is_trained 
        index_PQ.add(search_dataset)
        time_2 = time.time()
        time_PQ_con = time_2- time_1
        PQ_file.write('time_con M '+str(param_PQ[0])+ ' nbits ' + str(param_PQ[1]) + ' ' + str(time_PQ_con)+'\n')
        print('finish build PQ')


        for  query_path in  query_list[i]:
            query_dataset = np.load(query_path)
            query_name = query_path.split('/')[-1].split('_')[0]
            print('now processing ', dataset_name, ' ', query_name)

            for k in query_k_list:
                print('now computing k = ', k)
                time_1 = time.time()
                dis_truth, ID_truth = index.search(query_dataset, k)
                query_length = query_dataset.shape[0]
                time_2 = time.time()
                time_brute_sea = time_2 - time_1

                recall_record = np.zeros((query_length, 1))
                time_1 = time.time()
                dis_IVF, ID_IVF = index_IVF.search(query_dataset, k)
                time_2 = time.time()
                for i in range(query_length):
                    ground_truth = ID_truth[i, :]
                    search_result = ID_IVF[i, :]
                    recall = len(set(ground_truth) & set(search_result)) / len(set(ground_truth))
                recall = 0
                for j in range(query_length):
                    recall += recall_record[j, 0]
                recall_record[j, 0] = recall / query_length
                qps = query_length / (time2 - time1)
                IVFFlat_file.write(query_name + ' k '+str(k)+' recall '+str(recall)+' qps '+str(qps)+'\n')
                np.save(os.path.join(save_path, dataset_name, 'IVFFlat', query_name+'_'+str(k)+'_recall.npy'), recall_record)
                np.save(os.path.join(save_path, dataset_name, 'IVFFlat', query_name+'_'+str(k)+'_dis.npy'), dis_IVF)
                print('IVFFlat finish')


                recall_record = np.zeros((query_length, 1))
                time_1 = time.time()
                dis_IVFPQ, ID_IVFPQ = index_IVFPQ.search(query_dataset, k)
                time_2 = time.time()
                for i in range(query_length):
                    ground_truth = ID_truth[i, :]
                    search_result = ID_IVFPQ[i, :]
                    recall_record[j, 0] = len(set(ground_truth) & set(search_result)) / len(set(ground_truth))
                recall = 0
                for j in range(query_length):
                    recall += recall_record[j, 0]
                recall = recall / query_length
                qps = query_length / (time2 - time1)
                IVFPQ_file.write(query_name + ' k '+str(k)+' recall '+str(recall)+' qps '+str(qps)+'\n')
                np.save(os.path.join(save_path, dataset_name, 'IVFPQ', query_name+'_'+str(k)+'_recall.npy'), recall_record)
                np.save(os.path.join(save_path, dataset_name, 'IVFPQ', query_name+'_'+str(k)+'_dis.npy'), dis_IVFPQ)
                print('IVFPQ finish')
               

                recall_record = np.zeros((query_length, 1))
                time_1 = time.time()
                dis_HNSW, ID_HNSW = index_HNSW.search(query_dataset, k)
                time_2 = time.time()
                for i in range(query_length):
                    ground_truth = ID_truth[i, :]
                    search_result = ID_HNSW[i, :]
                    recall_record[j, 0] = len(set(ground_truth) & set(search_result)) / len(set(ground_truth))
                recall = 0
                for j in range(query_length):
                    recall += recall_record[j, 0]
                recall = recall / query_length
                qps = query_length / (time2 - time1)
                HNSW_file.write(query_name + ' k '+str(k)+' recall '+str(recall)+' qps '+str(qps)+'\n')
                np.save(os.path.join(save_path, dataset_name, 'HNSW', query_name+'_'+str(k)+'_recall.npy'), recall_record)
                np.save(os.path.join(save_path, dataset_name, 'HNSW', query_name+'_'+str(k)+'_dis.npy'), dis_HNSW)
                print('HNSW finish')

                
                recall_record = np.zeros((query_length, 1))
                time_1 = time.time()
                dis_LSH, ID_LSH = index_LSH.search(query_dataset, k)
                time_2 = time.time()
                for i in range(query_length):
                    ground_truth = ID_truth[i, :]
                    search_result = ID_LSH[i, :]
                    recall_record[j, 0] = len(set(ground_truth) & set(search_result)) / len(set(ground_truth))
                recall = 0
                for j in range(query_length):
                    recall += recall_record[j, 0]
                recall = recall / query_length
                qps = query_length / (time2 - time1)
                LSH_file.write(query_name + ' k '+str(k)+' recall '+str(recall)+' qps '+str(qps)+'\n')
                np.save(os.path.join(save_path, dataset_name, 'LSH', query_name+'_'+str(k)+'_recall.npy'), recall_record)
                np.save(os.path.join(save_path, dataset_name, 'LSH', query_name+'_'+str(k)+'_dis.npy'), dis_LSH)
                print('LSH finish')


                recall_record = np.zeros((query_length, 1))
                time_1 = time.time()
                dis_PQ, ID_PQ = index_PQ.search(query_dataset, k)
                time_2 = time.time()
                for i in range(query_length):
                    ground_truth = ID_truth[i, :]
                    search_result = ID_PQ[i, :]
                    recall_record[j, 0] = len(set(ground_truth) & set(search_result)) / len(set(ground_truth))
                recall = 0
                for j in range(query_length):
                    recall += recall_record[j, 0]
                recall = recall / query_length
                qps = query_length / (time2 - time1)
                PQ_file.write(query_name + ' k '+str(k)+' recall '+str(recall)+' qps '+str(qps)+'\n')
                np.save(os.path.join(save_path, dataset_name, 'PQ', query_name+'_'+str(k)+'_recall.npy'), recall_record)
                np.save(os.path.join(save_path, dataset_name, 'PQ', query_name+'_'+str(k)+'_dis.npy'), dis_PQ)
                print('PQ finish')





