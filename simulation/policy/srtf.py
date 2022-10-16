from .policy import Policy


class ShortestRemainingTimeFirst(Policy):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts):
        super(ShortestRemainingTimeFirst, self).__init__(trace, vc, placement, log_dir, logger, start_ts)
        self._name = "srtf"

    def simulate(self):
        prev_index = 0

        while self.end_job_num != self.total_job_num:

            """1. Check & Release End Jobs"""
            run_ls = self.run_list.copy()  # Avoid list.remove() issue
            for job in run_ls:
                if job["remain"] == 0:
                    job["status"] = "end"
                    job["end_time"] = self.time
                    self.end_job_num += 1
                    assert self._vc.release_resource(job) == True
                    self.run_list.remove(job)
                else:
                    job["remain"] -= 1

            """2. Check New Jobs """
            for idx in range(prev_index, self.total_job_num):
                job = self.trace.job_list[idx]
                if job["submit_time"] == self.time:
                    job["status"] = "pend"
                    self.que_list.append(job)
                    prev_index = idx
                elif job["submit_time"] > self.time:
                    break

            """3. Select Job to Preempt or Run """
            # NOTE: Sort by remain -- SRTF

            current_job = self.que_list + self.run_list
            current_job.sort(key=lambda x: x.__getitem__("remain"))

            quota = self._vc.total_gpus
            preempt_list = []
            prerun_list = []
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
