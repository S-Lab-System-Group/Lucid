from .policy import Policy


class Tiresias(Policy):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts):
        super(Tiresias, self).__init__(trace, vc, placement, log_dir, logger, start_ts)
        self._name = "tiresias"

        # Refer to https://github.com/SymbioticLab/Tiresias
        self._discretize_threshold = 18000
        self._low_priority_queue = []
        self._high_priority_queue = []

    def discretize_queue(self, job_queue):
        self._low_priority_queue = []
        self._high_priority_queue = []
        for job in job_queue:
            if job["priority"] > self._discretize_threshold:
                self._low_priority_queue.append(job)
            else:
                self._high_priority_queue.append(job)

        # Tiresias: Jobs in the same queue are scheduled in a FIFO order
        self._low_priority_queue.sort(key=lambda x: x.__getitem__("submit_time"))
        self._high_priority_queue.sort(key=lambda x: x.__getitem__("submit_time"))

    def simulate(self):
        prev_index = 0

        while self.end_job_num != self.total_job_num:

            """1. Check & Release End Jobs"""
            run_ls = self.run_list.copy()
            for job in run_ls:
                if job["remain"] == 0:
                    job["status"] = "end"
                    job["end_time"] = self.time
                    self.end_job_num += 1
                    assert self._vc.release_resource(job) == True
                    self.run_list.remove(job)
                else:
                    job["remain"] -= 1
                    job["priority"] += job.__getitem__("gpu_num")

            """2. Check New Jobs """
            for idx in range(prev_index, self.total_job_num):
                job = self.trace.job_list[idx]
                if job["submit_time"] == self.time:
                    job["status"] = "pend"
                    job["priority"] = 0
                    self.que_list.append(job)
                    prev_index = idx
                elif job["submit_time"] > self.time:
                    break

            """3. Select Job to Preempt or Run """
            preempt_list = []
            prerun_list = []
            # Refer to Pollux implementation, scheduling interval = 60s by default
            if self.time % 60 == 0:
                current_job = self.run_list + self.que_list
                quota = self._vc.total_gpus
                self.discretize_queue(current_job)
                current_job = self._high_priority_queue + self._low_priority_queue

                for job in current_job:
                    if job.__getitem__("gpu_num") <= quota:
                        quota -= job.__getitem__("gpu_num")
                        if job["status"] == "pend":
                            prerun_list.append(job)
                    elif job["status"] == "run":
                        preempt_list.append(job)

            """4. Preempt Job """
            for job in preempt_list:
                job["ckpt_times"] += 1
                job.set_ckpt_time(self.time)
                job["status"] = "pend"
                job["remain"] += self.ckpt_overhead(job)
                assert self._vc.release_resource(job) == True
                job["nodes"] = []

                if job not in self.que_list:
                    self.que_list.append(job)
                if job in self.run_list:
                    self.run_list.remove(job)

            """5. Allocate Job """
            for job in prerun_list:
                if self.job_placer(job):
                    job["status"] = "run"
                    if job["ckpt_times"] == 0:
                        job["start_time"] = self.time
                        job["queue"] = self.time - job["submit_time"]
                    else:
                        job["queue"] = job["queue"] + (self.time - job.get_ckpt_time())

                    if job in self.que_list:
                        self.que_list.remove(job)
                    if job not in self.run_list:
                        self.run_list.append(job)
                else:
                    # May place fail because consolidate requirement
                    if job not in self.que_list:
                        self.que_list.append(job)
                    continue

            """6. Log & Result Recorder"""
            if self.time % 10000 == 0:
                self.runtime_log()

            # Sample Cluster State Every Minute
            if self.time % 60 == 0:
                self.seq_recorder()

            self.time += 1

        self.log_recorder(self._name)
