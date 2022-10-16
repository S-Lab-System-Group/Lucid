from .profiler import Profiler


class LeastGPUFirstProfiler(Profiler):
    def __init__(self, trace, scale, time_limit, prof_gpu_limit, placement, log_dir, logger, start_ts):
        super(LeastGPUFirstProfiler, self).__init__(
            trace, scale, time_limit, prof_gpu_limit, placement, log_dir, logger, start_ts
        )
        self._name = "lgfprof"
        self.cluster_name = log_dir.split("/")[-1].split("_")[0]
        self.get_time_series_data(self.cluster_name)
        self.enable_scaling = True if self.cluster_name == "Venus" else False
        self.node_scaling_time = 0
        self.node_scaling_num = 1

    def profile(self):
        prev_index = 0

        while self.end_job_num != self.total_job_num:

            """1. Check & Release End Jobs"""
            run_ls = self.run_list.copy()  # Avoid list.remove() issue
            for job in run_ls:
                if self.time == job["end_time"]:
                    if job["toskip"] == 1:
                        job["remain"] = 0
                        job["status"] = "end"
                    self.end_job_num += 1
                    assert self._vc.release_resource(job)
                    self.run_list.remove(job)

            """2. Allocate New / Pending Jobs"""
            # New Job
            for idx in range(prev_index, self.total_job_num):
                job = self.trace.job_list[idx]
                if job["gpu_num"] > self.gpu_limit:
                    self.end_job_num += 1
                    prev_index = idx + 1
                else:
                    if job["submit_time"] == self.time:
                        self.que_list.append(job)
                        prev_index = idx
                    elif job["submit_time"] > self.time:
                        break

            # Pend Job
            # NOTE: Sort by Job GPU Num -- LGF
            self.que_list.sort(key=lambda x: x.__getitem__("gpu_num"))
            # self.que_list.sort(key=lambda x: x.__getitem__("submit_time"))
            que_ls = self.que_list.copy()
            for job in que_ls:
                if self.job_placer(job):
                    job["profiled"] = 1
                    job["start_time"] = self.time
                    job["profqueue"] = self.time - job["submit_time"]
                    job["queue"] = job["profqueue"]
                    if job["duration"] <= self.time_limit:
                        job["end_time"] = job["start_time"] + job["duration"]
                        job["toskip"] = 1
                    else:
                        job["end_time"] = job["start_time"] + self.time_limit
                    self.que_list.remove(job)
                    self.run_list.append(job)
                else:
                    break

            """3. Time-aware Scaling (Optional)"""
            if self.enable_scaling:
                # Scale-Up
                if self.time % 10 == 0 and len(self.que_list) > 10 and self._vc.node_num == self._vc.base_node_num:
                    self._vc.update_vc_node(change_node_num=self.node_scaling_num)
                    self.node_scaling_time = self.time
                    self.scaling_recorder(self.node_scaling_num)

                # Scale-Down
                if (
                    self.time % 100 == 0
                    and len(self.que_list) < 5
                    and self._vc.node_num == self._vc.base_node_num + self.node_scaling_num
                    and len(self._vc.idle_node_list()) >= self.node_scaling_num
                    and self._vc.check_node_inside_idle_vc(self._vc.temp_node_num_base)
                ):
                    if self.check_future_cluster_throughput() <= self.gpu_limit * 5:
                        self._vc.update_vc_node(change_node_num=-1 * self.node_scaling_num)
                        self.node_scaling_time = self.time
                        self.scaling_recorder(-1 * self.node_scaling_num)

            """4. Log & Result Recorder"""
            if self.time % 10000 == 0:
                self.runtime_log()

            # Sample Cluster State Every Minute
            if self.time % 60 == 0:
                self.seq_recorder()

            self.time += 1

        self.log_recorder(self._name)
