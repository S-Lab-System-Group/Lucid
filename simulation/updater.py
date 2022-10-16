class ColocateUpdater:
    def __init__(self, colocate_df):
        self.df = colocate_df

    def _query(self, job1, job2):
        reverse = 0
        m1, m2 = job1["model"], job2["model"]
        d1, d2 = job1["dataset"], job2["dataset"]
        b1, b2 = job1["batchsize"], job2["batchsize"]
        a1, a2 = job1["amp"], job2["amp"]
        # g1, g2 = job1["gpu_num"], job2["gpu_num"]  # NOTE

        info = self.df.query(
            " model1 == @m1 and model2 == @m2 and batchsize1 == @b1 and batchsize2 == @b2 and dataset1 == @d1 and dataset2 == @d2 and amp1 == @a1 and amp2 == @a2"
        )
        if len(info) == 0:
            info = self.df.query(
                " model1 == @m2 and model2 == @m1 and batchsize1 == @b2 and batchsize2 == @b1 and dataset1 == @d2 and dataset2 == @d1 and amp1 == @a2 and amp2 == @a1"
            )
            reverse = 1
        assert len(info) == 1, f"job1: {job1} | job2: {job2}"
        return info, reverse

    def query_info(self, job1, job2):
        if self.check_outside_job(job1, job2):
            # Little Influence
            total_util = min(1, job1["gpu_util"] + job2["gpu_util"])
            total_mem = job1["gmem"] + job2["gmem"]
            return 1, 1, total_util, total_mem
        else:
            info, reverse = self._query(job1, job2)
        speed1, speed2 = info["speed1"].values[0], info["speed2"].values[0]
        if reverse:
            return speed2, speed1, info["gpu_util"].values[0], info["gmem"].values[0]
        else:
            return speed1, speed2, info["gpu_util"].values[0], info["gmem"].values[0]

    def query_speed(self, job1, job2):
        if self.check_outside_job(job1, job2):
            # Little Influence
            return 1, 1
        else:
            info, reverse = self._query(job1, job2)
        speed1, speed2 = info["speed1"].values[0], info["speed2"].values[0]
        if reverse:
            return speed2, speed1
        else:
            return speed1, speed2

    def query_utils(self, job1, job2):

        if self.check_outside_job(job1, job2):
            # Approximate as adding
            total_util = min(1, job1["gpu_util"] + job2["gpu_util"])
            total_mem = job1["gmem"] + job2["gmem"]
            return total_util, total_mem
        else:
            info, _ = self._query(job1, job2)
        return info["gpu_util"].values[0], info["gmem"].values[0]

    # Some Jobs are not recorded inside colocate_df
    def check_outside_job(self, job1, job2):
        m1, m2 = job1["model"], job2["model"]
        models = [m1, m2]
        if "NeuMF" in models:
            return True
        # Large Model are classified as 2
        elif "ResNet50" in models or "BERT" in models or "Transformer" in models:
            # raise NotImplementedError
            return True
        else:
            return False
