import sys
import pandas as pd
import time
from multiprocessing import Process, Manager, Value

from smi import smi_getter


def s_collect(fun, dataset, model_name, batch_size, mp, gpu_id):
    with Manager() as manager:
        smi_list = manager.list()
        speed_list = manager.list()
        warm_signal = Value('i', 0)

        p1 = Process(target=fun, args=(model_name, batch_size, mp, gpu_id, speed_list, warm_signal, ))
        p2 = Process(target=smi_getter, args=(sys.argv[1:], smi_list, gpu_id, ))

        t_begin = time.time()
        p1.start()
        while True:
            if warm_signal.value == 1:
                p2.start()
                break
        
        p1.join()
        p2.terminate()
        t_pass = time.time() - t_begin

        speed_list = list(speed_list)
        smi_df = pd.DataFrame(list(smi_list))
        smi_df.drop([0])
        smi_df.drop([len(smi_df)-1], inplace=True)
        # print(smi_df)
        
        metric_dict = {'model': model_name, 'dataset': dataset, 'gpu_num': len(gpu_id), 'batchsize': batch_size, 'amp': mp}
        metric_dict['speed'] = round(speed_list[0], 3)

        # Process gpu info
        metric_dict['gpu_util'] = round(pd.to_numeric(smi_df['gpuUtil']).mean(), 3)
        metric_dict['gmem_util'] = round(pd.to_numeric(smi_df['gpuMemUtil']).mean(), 3)
        smi_df['gpuMem'] = smi_df['gpuMem'].apply(lambda x: x[:-4]).astype('int64')
        metric_dict['gmem'] = round(smi_df['gpuMem'].mean(), 3)
        metric_dict['time'] = t_pass
        
    return metric_dict



