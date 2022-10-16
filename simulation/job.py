import sys


class Job(dict):
    def __init__(self, series):
        super(Job, self).__init__()
        self.update(series.to_dict())
        # Priority Define by Estimator, Random Means No History Data Found
        self.update({"nodes": [], "priority": -1, "random": 0})
        # Profiler
        self.update({"profiled": 0, "profqueue": 0, "toskip": 0})
        # Co-locate
        # NOTE: exclusive: {0: colocate, 1: exclusive}
        # NOTE: rate: the ratio of colocate and exclusive execution performance
        # NOTE: sharescore: 0, 1, 2
        self.update({"exclusive": 1, "rate": 1, "sharescore": None, "Tcolocate": 0, "Tdelocate": 0})

    def set_ckpt_time(self, time):
        self.last_ckpt_time = time

    def get_ckpt_time(self):
        return self.last_ckpt_time


class Trace:
    def __init__(self):
        self.job_list = []

    def append_job(self, job):
        self.job_list.append(job)

    def job_num(self):
        return len(self.job_list)

    def profiler_remain_job_num(self):
        num = 0
        for job in self.job_list:
            if job["toskip"] == 0:
                num += 1
        return num

    def sort_jobs(self, key):
        self.job_list.sort(key=lambda x: x.__getitem__(key))

    def vc_trace(self, vc_name):
        vc_trace = Trace()
        for job in self.job_list:
            if job["vc"] == vc_name:
                vc_trace.append_job(job)
        vc_trace.sort_jobs("submit_time")
        return vc_trace

    def reset_trace(self):
        for job in self.job_list:
            if job["toskip"] == 0:
                job["start_time"] = sys.maxsize
                job["end_time"] = sys.maxsize
                job["nodes"] = []
