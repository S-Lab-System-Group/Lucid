from .policy import Policy


class ShortestJobFirst(Policy):
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts):
        super(ShortestJobFirst, self).__init__(trace, vc, placement, log_dir, logger, start_ts)
        self._name = "sjf"

    def simulate(self):
        prev_index = 0

        while self.end_job_num != self.total_job_num:

            """1. Check & Release End Jobs"""
            run_ls = self.run_list.copy()  # Avoid list.remove() issue
            for job in run_ls:
                if self.time == job["end_time"]:
                    job["remain"] = 0
                    job["status"] = "end"
                    self.end_job_num += 1
                    assert self._vc.release_resource(job) == True
                    self.run_list.remove(job)

            """2. Allocate New / Pending Jobs"""
            # New Job
            for idx in range(prev_index, self.total_job_num):
                job = self.trace.job_list[idx]
                if job["submit_time"] == self.time:
                    job["status"] = "pend"
                    self.que_list.append(job)
                    prev_index = idx
                elif job["submit_time"] > self.time:
                    break

            # Pend Job
            # NOTE: Sort by duration -- SJF
            self.que_list.sort(key=lambda x: x.__getitem__("duration"))
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

            """3. Log & Result Recorder"""
            if self.time % 10000 == 0:
                self.runtime_log()

            # Sample Cluster State Every Minute
            if self.time % 60 == 0:
                self.seq_recorder()

            self.time += 1

        self.log_recorder(self._name)
