import random
import operator
from .policy import Policy


GPU_MEMORY_LIMITATION = 24576  # RTX 3090 24GB Memory for our benchmarking


class Lucid(Policy):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts, estimator, updater):
        super(Lucid, self).__init__(trace, vc, placement, log_dir, logger, start_ts)
        self.estimator = estimator
        self.updater = updater
        self._name = "lucid"

        self.enable_colocate()
        self.adaptive_colocate = 0
        self.obtain_workload_estimates()
        self.obtain_colocate_analysis()
        self.obtain_cluster_prediction()

    def enable_colocate(self):
        self._vc.colocate_enable = 1

    def obtain_workload_estimates(self):
        estimate = self.estimator.data
        for job in self.trace.job_list:
            if job["toskip"] == 0:
                job["priority"] = estimate[estimate["job_id"] == job["job_id"]]["priority"].iloc[0] * job["gpu_num"]

    def obtain_colocate_analysis(self):
        self.get_colocate_data()
        df = self.colo_df
        for job in self.trace.job_list:
            if job["toskip"] == 0:
                m, b, d, a = job["model"], job["batchsize"], job["dataset"], job["amp"]
                info = df.query(" model == @m and batchsize == @b and dataset == @d and amp == @a")
                job["sharescore"] = info["label"].values[0]

    def obtain_cluster_prediction(self):
        cluster = self.estimator.cluster_name
        self.get_time_series_data(cluster)
        if cluster == "Venus":
            self.get_profile_scaling_data(cluster)

    def obtain_job_from_id(self, id):
        for job in self.run_list:
            if job["job_id"] == id:
                return job

    # Prescient Adaptive Sharing
    def check_pas(self):
        if self.estimator.cluster_name == "Venus" and self.check_future_cluster_throughput() <= 2:
            return 0
        else:
            return 1

        # return 1

    def colocate_update(self, job, target_job):
        speed1, speed2, gutil, gmem = self.updater.query_info(job, target_job)
        job["exclusive"], target_job["exclusive"] = 0, 0
        job["rate"], target_job["rate"] = speed1, speed2
        job["Tcolocate"] = self.time
        return gutil, gmem

    def speed_recover(self, job_list):
        if isinstance(job_list, list):
            for job in job_list:
                job["exclusive"] = 1
                job["rate"] = 1
                job["Tdelocate"] = self.time
        else:
            job_list["exclusive"] = 1
            job_list["rate"] = 1
            job_list["Tdelocate"] = self.time

    def ablation_picker(self, job):
        mem_limit = GPU_MEMORY_LIMITATION - job["gmem"]
        affinity_jobs = []
        for j in self.run_list:
            if j["exclusive"] == 0:
                continue
            if j["gpu_num"] == job["gpu_num"] and j["gmem"] < mem_limit:
                affinity_jobs.append(j)

        if affinity_jobs:
            return affinity_jobs[0]
        else:
            return False

    def job_pair_picker_time_aware(self, job):
        mem_limit = GPU_MEMORY_LIMITATION - job["gmem"]
        affinity_jobs = []
        for j in self.run_list:
            if j["exclusive"] == 0:
                continue
            if j["gpu_num"] == job["gpu_num"] and j["gmem"] < mem_limit and (job["sharescore"] + j["sharescore"]) <= 2:
                if job["priority"] <= j["priority"] * 2:
                    affinity_jobs.append(j)

        if affinity_jobs:
            # if job["sharescore"] == 0 or job["sharescore"] == 1:
            #     affinity_jobs.sort(key=lambda x: x.__getitem__("sharescore"))
            #     return affinity_jobs[0]
            # else:
            #     return affinity_jobs[0]
            return affinity_jobs[0]
            # return random.choice(affinity_jobs)
        else:
            return False

    def job_allocate_info_update(self, job):
        job["start_time"] = self.time
        job["queue"] = job["queue"] + self.time - job["submit_time"]
        job["status"] = "run"
        self.que_list.remove(job)
        self.run_list.append(job)

    def simulate(self):
        prev_index = 0
        stale_que = []

        while self.end_job_num != self.total_job_num:
            new_job_num = 0

            """1. Check & Release End Jobs"""
            run_ls = self.run_list.copy()  # Avoid list.remove() issue
            for job in run_ls:
                if job["remain"] <= 0:
                    job["status"] = "end"
                    job["end_time"] = self.time
                    self.end_job_num += 1
                    if self._vc.colocate_enable and job["exclusive"] == 0:
                        colo_job_id = self._vc.check_vc_colocate_jobs(job)
                        if colo_job_id:
                            colo_job = self.obtain_job_from_id(colo_job_id)
                            self.speed_recover(colo_job)

                    self._vc.release_resource(job)
                    self.run_list.remove(job)
                    # if self.estimator.name != "LGBEstimator" and self.estimator.name != "PhillyEstimator":
                    #     self.estimator.update_train_data(job)
                else:
                    job["remain"] -= job["rate"]

            """2. Check New Jobs"""
            # New Job
            for idx in range(prev_index, self.total_job_num):
                job = self.trace.job_list[idx]
                if job["toskip"]:
                    prev_index = idx + 1
                    self.end_job_num += 1
                    continue

                if job["submit_time"] == self.time:
                    job["status"] = "pend"
                    self.que_list.append(job)
                    prev_index = idx
                    new_job_num += 1
                elif job["submit_time"] > self.time:
                    break

            """3. Sort Job According to Priority"""
            self.que_list.sort(key=lambda x: x.__getitem__("priority"))

            """4. Allocate Job"""
            que_ls = self.que_list.copy()
            if self.time % 100 == 0:
                self.adaptive_colocate = self.check_pas()
            if self.adaptive_colocate == 0:  # Disable colocation
                for job in que_ls:
                    if self.job_placer(job):
                        self.job_allocate_info_update(job)
                    else:
                        break
            else:
                for job in que_ls:
                    if job["gpu_num"] <= 8:
                        # target_job = self.ablation_picker(job)
                        target_job = self.job_pair_picker_time_aware(job)
                        if target_job:
                            gutil, gmem = self.colocate_update(job, target_job)
                            assert self.colocate_job_placer(job, target_job, gutil, gmem)
                            self.job_allocate_info_update(job)
                        else:
                            if self.job_placer(job):
                                self.job_allocate_info_update(job)
                    else:
                        if self.job_placer(job):
                            self.job_allocate_info_update(job)

            """Echo Profiler Scaling"""
            if self.vc_echo_scaling and self.time % 10 == 0 and self.time in self.scaling_time_list:
                scaling_num = (
                    -1 * self.profile_scaling_df[self.profile_scaling_df["time"] == self.time]["scaling_num"].values[0]
                )
                self.logger.info(f"Time: {self.time}, Scaling Num: {scaling_num}")
                self._vc.update_vc_node(change_node_num=scaling_num)

            """5. Log & Result Recorder"""
            if self.time % 10000 == 0:
                self.runtime_log()

            # Sample Cluster State Every Minute
            if self.time % 60 == 0:
                self.seq_recorder()

            self.time += 1

        self.log_recorder(self._name)
