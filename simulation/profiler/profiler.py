import sys

sys.path.append("..")

from cluster import VC
from policy.placer.consolidate import ConsolidatePlacement
from policy.placer.consolidateFirst import ConsolidateFirstPlacement
from policy.placer.random import RandomPlacement

import pandas as pd
import os


class Profiler:
    def __init__(self, trace, scale, time_limit, prof_gpu_limit, placement, log_dir, logger, start_ts):
        self.trace = trace
        self.logger = logger
        self.time = start_ts
        self.time_limit = time_limit
        self.scale = scale
        self._placement = placement
        self._log_dir = log_dir
        self.prof_gpu_limit = prof_gpu_limit
        self.start_ts = start_ts

        self._vc = self.init_prof_vc()
        self._vc_name = self._vc.vc_name
        self.gpu_limit = self.set_gpu_limit()

        self.total_job_num = self.trace.job_num()
        self.que_list = []  # Pending Jobs
        self.run_list = []  # Running Jobs
        self.end_job_num = 0

        # Time Sequence Recorder
        self.time_list = []
        self.total_gpu_num = []
        self.idle_gpu_num = []
        self.pend_gpu_num = []
        self.run_job_num = []
        self.pend_job_num = []
        self.pend_job_num_less_8 = []
        self.total_node_num = []
        self.consolidate_node_num = []
        self.shared_node_num = []

        # Scaling Recorder
        self.scaling_time_list = []
        self.scaling_num_list = []

        self.placer = self.init_placer()

    def set_gpu_limit(self):
        return int(self.scale * self.prof_gpu_limit)

    def init_prof_vc(self):
        return VC("profvc", self.scale, 8, 96)

    def init_placer(self):
        if self._placement == "consolidate":
            return ConsolidatePlacement(self._vc)
        if self._placement == "random":
            return RandomPlacement(self._vc)
        if self._placement == "consolidateFirst":
            return ConsolidateFirstPlacement(self._vc)
        raise NotImplementedError

    def get_time_series_data(self, cluster):
        self.time_df = pd.read_csv(f"predictor/{cluster}_throughput_pred.csv", parse_dates=["time"])

        self.time_df["time"] = self.time_df["time"] - pd.Timestamp(self.time_df["time"][0])
        self.time_df["time"] = self.time_df["time"].map(lambda x: x.seconds + 3600 * 24 * x.days)
        self.time_df["time"] = self.time_df["time"] + self.start_ts
        # # self.time_df = self.time_df.set_index("time")

    def check_future_cluster_throughput(self):
        if len(self.time_df) == 0:
            return -1
        else:
            self.time_df = self.time_df[self.time_df["time"] > self.time]
            if len(self.time_df) >= 6:
                return self.time_df.head()["pred_gpu_num"].mean()
            else:
                return self.time_df["pred_gpu_num"].mean()

        # future_pred = self.time_df[self.time_df["time"] > self.time]
        # if len(future_pred) >= 6:
        #     future_pred = future_pred[:6]
        # elif len(future_pred) == 0:
        #     return -1
        # return future_pred["pred_gpu_num"].mean()

    def job_placer(self, job):
        return self.placer.place(job)

    def runtime_log(self):
        self.logger.info(
            f"{self._vc_name} | Time: {int(self.time)} | Total Job: {self.total_job_num} | End job: {self.end_job_num} | Running job: {len(self.run_list)} | Pending job: {len(self.que_list)} | Avail Nodes: {len(self._vc.avail_node_list())}"
        )

    """Simulation Result Recorder"""

    def log_recorder(self, policy_name):
        if not os.path.exists(os.path.join(self._log_dir, self._vc_name)):
            os.mkdir(os.path.join(self._log_dir, self._vc_name))

        df = pd.DataFrame(self.trace.job_list)

        if len(df) == 0:
            print("No Job in VC: ", self._vc_name)
            raise NotImplementedError

        sum_prof = df["profiled"].sum()
        frac_prof = round(sum_prof / self.total_job_num * 100, 3)
        sum_skip = df["toskip"].sum()
        frac_skip = round(sum_skip / self.total_job_num * 100, 3)
        avg_que = round(df["profqueue"].mean(), 2)

        self.logger.info(
            f"{self._vc_name} | Profiled Job Num: {sum_prof} ({frac_prof}%) | Finish in Profile: {sum_skip} ({frac_skip}%) | Average Queue: {avg_que}"
        )

        path = f"{self._log_dir}/logfile/{self._vc_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        df.to_csv(
            f"{path}/{policy_name}_{self._placement}_time{self.time_limit}_scale{self.scale}_factor{self.prof_gpu_limit}_log.csv",
            index=False,
        )

        # Time Sequence
        seq_dict = {
            "time": self.time_list,
            "total_gpu_num": self.total_gpu_num,
            "idle_gpu_num": self.idle_gpu_num,
            "pending_gpu_num": self.pend_gpu_num,
            "running_gpujob_num": self.run_job_num,
            "pending_gpujob_num": self.pend_job_num,
            "pending_job_num_less_8": self.pend_job_num_less_8,
            "total_node_num": self.total_node_num,
            "consolidate_node_num": self.consolidate_node_num,
            "shared_node_num": self.shared_node_num,
        }
        seq = pd.DataFrame(seq_dict)
        seq["gpu_utilization"] = ((seq["total_gpu_num"] - seq["idle_gpu_num"]) / seq["total_gpu_num"]).round(3)
        seq.to_csv(f"{self._log_dir}/{self._vc_name}/{policy_name}_{self._placement}_{self._vc_name}_seq.csv", index=False)

        # Scaling Sequence
        scaling_dict = {"time": self.scaling_time_list, "scaling_num": self.scaling_num_list}
        scaling = pd.DataFrame(scaling_dict)
        scaling.to_csv(f"{self._log_dir}/{self._vc_name}/{self._vc_name}_scaling.csv", index=False)

    def pend_job_num_small(self):
        job_num = 0
        for job in self.que_list:
            if job.__getitem__("gpu_num") < 8:
                job_num += 1
        return job_num

    def seq_recorder(self):
        self.time_list.append(self.time)
        self.total_gpu_num.append(self._vc.total_gpus)
        self.idle_gpu_num.append(self._vc.vc_free_gpus())
        self.pend_gpu_num.append(sum(job.__getitem__("gpu_num") for job in self.que_list))
        self.run_job_num.append(len(self.run_list))
        self.pend_job_num.append(len(self.que_list))
        self.pend_job_num_less_8.append(self.pend_job_num_small())
        self.total_node_num.append(self._vc.node_num)
        self.consolidate_node_num.append(self._vc.consolidate_node_num())
        self.shared_node_num.append(self._vc.shared_node_num())

    def scaling_recorder(self, num):
        self.scaling_time_list.append(self.time)
        self.scaling_num_list.append(num)
