from .policy import Policy


class QuasiShortestServiceFirst(Policy):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts, estimator):
        super(QuasiShortestServiceFirst, self).__init__(trace, vc, placement, log_dir, logger, start_ts)
        self.estimator = estimator
        self._name = "qssf"

    def simulate(self):
        prev_index = 0

        while self.end_job_num != self.total_job_num:
            new_job_num = 0

            """1. Check & Release End Jobs"""
            run_ls = self.run_list.copy()  # Avoid list.remove() issue
            for job in run_ls:
                if self.time == job["end_time"]:
                    job["remain"] = 0
                    job["status"] = "end"
                    self.end_job_num += 1
                    assert self._vc.release_resource(job) == True
                    self.run_list.remove(job)
                    if self.estimator.name != "LGBEstimator" and self.estimator.name != "PhillyEstimator":
                        self.estimator.update_train_data(job)

            """2. Check New Jobs"""
            # New Job
            for idx in range(prev_index, self.total_job_num):
                job = self.trace.job_list[idx]
                if job["submit_time"] == self.time:
                    job["status"] = "pend"
                    self.que_list.append(job)
                    prev_index = idx
                    new_job_num += 1
                elif job["submit_time"] > self.time:
                    break

            """3. Assign Priority If Exist Job Pending"""
            # NOTE: Sort by priority given by estimator -- QSSF
            # Only assign priority to the pending job, new job will sort by required gpu_num
            self.que_list.sort(key=lambda x: x.__getitem__("gpu_num"))
            if len(self.que_list) > new_job_num:
                for job in self.que_list:
                    if job["priority"] == -1:
                        job["priority"] = self.estimator.inference(job)
                self.que_list.sort(key=lambda x: x.__getitem__("priority"))

            """4. Allocate Job"""
            que_ls = self.que_list.copy()  # Avoid list.remove() issue
            for job in que_ls:
                if self.job_placer(job):
                    job["start_time"] = self.time
                    job["end_time"] = job["start_time"] + job["duration"]
                    job["queue"] = self.time - job["submit_time"]
                    job["status"] = "run"
                    self.que_list.remove(job)
                    self.run_list.append(job)
                else:
                    break

            """5. Log & Result Recorder"""
            if self.time % 10000 == 0:
                self.runtime_log()

            # Sample Cluster State Every Minute
            if self.time % 60 == 0:
                self.seq_recorder()

            self.time += 1

        self.log_recorder(self._name)
