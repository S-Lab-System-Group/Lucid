import sys
import pandas as pd
import time
from multiprocessing import Process, Manager, Value

from smi import smi_getter


def collect(fun1, m_list1, dataset1, bs_list1, fun2, m_list2, dataset2, bs_list2, gpu_id):
    metric_list = []
    if dataset1 == 'LunarLander' or dataset1 == 'BipedalWalker' or dataset1 == 'Multi30k':
        mp_list1 = [0]
    else:
        mp_list1 = [0, 1]

    if dataset2 == 'LunarLander' or dataset2 == 'BipedalWalker' or dataset1 == 'Multi30k':
        mp_list2 = [0]
    else:
        mp_list2 = [0, 1]
    for model_name1 in m_list1:
        for model_name2 in m_list2:
            for batch_size1 in bs_list1:
                for batch_size2 in bs_list2: 
                    if model_name1 == 'resnet50':
                        batch_size1 = 32
                    if model_name2 == 'resnet50':
                        batch_size2 = 32
                    for mp1 in mp_list1:
                        for mp2 in mp_list2:
                            # Check whether there are duplicated pairs
                            df_check = pd.DataFrame(metric_list, columns=['model1', 'dataset1', 'gpu_num1', 'batchsize1', 'amp1', 'speed1', 'model2', 'dataset2', 'gpu_num2', 'batchsize2', 'amp2', 'speed2', 'gpu_util', 'gmem_util', 'gmem'])
                            # print(df_check[['model1', 'batchsize1', 'amp1', 'model2', 'batchsize2', 'amp2']])

                            info = df_check.query(" model1 == @model_name2 and model2 == @model_name1 and batchsize1 == @batch_size2 and batchsize2 == @batch_size1 and dataset1 == @dataset2 and dataset2 == @dataset1 and amp1 == @mp2 and amp2 == @mp1")
                            if not info.empty:
                                continue
                            info2 = df_check.query(" model1 == @model_name1 and model2 == @model_name2 and batchsize1 == @batch_size1 and batchsize2 == @batch_size2 and dataset1 == @dataset1 and dataset2 == @dataset2 and amp1 == @mp1 and amp2 == @mp2")
                            if not info2.empty:
                                continue

                            # collect co-locate jobs gpu info
                            print('co-locate:')
                            with Manager() as manager:
                                smi_list = manager.list()
                                speed_list1 = manager.list()
                                speed_list2 = manager.list()
                                signal1 = Value('i', 0)
                                signal2 = Value('i', 0)

                                p1 = Process(target=fun1, args=(model_name1, batch_size1, mp1, gpu_id, speed_list1, signal1, ))
                                p2 = Process(target=fun2, args=(model_name2, batch_size2, mp2, gpu_id, speed_list2, signal2, ))
                                p3 = Process(target=smi_getter, args=(sys.argv[1:], smi_list, gpu_id, ))

                                p1.start()
                                p2.start()
                                while True:
                                    if signal1.value == 1 and signal2.value == 1:
                                        p3.start()
                                        break
                                
                                p1.join()
                                p2.join()
                                p3.terminate()
                            
                                speed_list1 = list(speed_list1)
                                speed_list2 = list(speed_list2)
                                smi_df = pd.DataFrame(list(smi_list))
                                smi_df.drop([0])
                                smi_df.drop([len(smi_df)-1], inplace=True)

                                d1 = {'model1': model_name1, 'dataset1': dataset1, 'gpu_num1': len(gpu_id), 'batchsize1': batch_size1, 'amp1': mp1}
                                d1['speed1'] = round(speed_list1[0], 3)
                                d2 = {'model2': model_name2, 'dataset2': dataset2, 'gpu_num2': len(gpu_id), 'batchsize2': batch_size2, 'amp2': mp2}
                                d2['speed2'] = round(speed_list2[0], 3)
                                metric_dict = {}
                                metric_dict.update(d1)
                                metric_dict.update(d2)
                                # Process gpu info
                                smi_df['gpuUtil'] = pd.to_numeric(smi_df['gpuUtil'])
                                metric_dict['gpu_util'] = round(pd.to_numeric(smi_df['gpuUtil']).mean(), 3)
                                metric_dict['gmem_util'] = round(pd.to_numeric(smi_df['gpuMemUtil']).mean(), 3)
                                smi_df['gpuMem'] = smi_df['gpuMem'].apply(lambda x: x[:-4]).astype('int64')
                                metric_dict['gmem'] = round(smi_df['gpuMem'].max(), 3)
                                
                                # print(metric_dict)
                                metric_list.append(metric_dict)
                            time.sleep(2)

    return metric_list