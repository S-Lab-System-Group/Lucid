from tkinter.messagebox import NO
import pandas as pd
import difflib
import random
from estimator import utils

random.seed(1)


class NaiveEstimator:
    def __init__(self, args):
        if "Sept" in args.experiment_name:
            trace_range = ("2020-09-01 00:00:00", "2020-09-26 23:59:00")
        elif "July" in args.experiment_name:
            trace_range = ("2020-07-01 00:00:00", "2020-07-31 23:59:00")
        else:
            raise ValueError
        self.train_data, _ = utils.train_data_loader(args.trace_dir, trace_range)
        self.job_names = self.train_data["jobname"].unique()
        self.name = "NaiveEstimator"

        """Hyperparameters"""
        self.match_num = 2
        # Generate Random Priority for New User
        self.new_user_range = (100, 20000)
        self.agg_method = "mean"  # Mean or Median

    def update_train_data(self, job):
        job = pd.DataFrame(job)
        job = job.loc[:, "job_id":"duration"]
        # self.train_data = self.train_data.append(job, ignore_index=True)
        self.train_data = pd.concat([self.train_data, job], ignore_index=True)
        self.job_names = self.train_data["jobname"].unique()

    def check_history_job(self, job, user_data):
        gpu_data = user_data[user_data["gpu_num"] == job["gpu_num"]]

        # Simple Mean / Median
        if len(gpu_data) > 0:
            user_data = gpu_data
        return int(user_data["duration"].agg(self.agg_method) * job["gpu_num"])

        # # Weighted Windows  (Only recent 3 jobs)
        # if len(gpu_data) > 0:
        #     user_data = gpu_data
        # user_data = user_data.sort_values(by='submit_time')
        # if len(user_data) >= 3:
        #     user_data = user_data.iloc[-3:]
        #     job['random'] = user_data['job_id'].iloc[-1]
        # return int(user_data['duration'].agg(self.agg_method) * job['gpu_num'])

    def inference(self, job):
        matches = difflib.get_close_matches(job["jobname"], self.job_names, n=self.match_num, cutoff=0.6)
        user_data = self.train_data[self.train_data["user"] == job["user"]]

        if len(matches) > 0:
            if matches[0] == job["jobname"]:
                user_data = user_data[user_data["jobname"] == job["jobname"]]
                if len(user_data) == 0:
                    return (
                        self.train_data[self.train_data["jobname"] == job["jobname"]]["duration"].agg(self.agg_method)
                        * job["gpu_num"]
                    )
                else:
                    return self.check_history_job(job, user_data)
            else:
                user_data = user_data[user_data["jobname"].isin(matches)]
                if len(user_data) == 0:
                    return (
                        self.train_data[self.train_data["jobname"].isin(matches)]["duration"].agg(self.agg_method)
                        * job["gpu_num"]
                    )
                else:
                    return self.check_history_job(job, user_data)
        else:
            job["random"] = 1  # estimate through randint/other job
            # return random.randint(100, 20000) * job['gpu_num']
            if len(user_data) == 0:  # new user
                return self.train_data["duration"].agg(self.agg_method) * job["gpu_num"]
            else:
                return user_data["duration"].agg(self.agg_method) * job["gpu_num"]


class LGBEstimator:
    def __init__(self, args):
        self.data = pd.read_csv(f"./estimator/{args.experiment_name}_lgb.csv")
        self.args = args
        self.name = "LGBEstimator"

    def inference(self, job):
        p = self.data[self.data["job_id"] == job["job_id"]]["priority"].iloc[0]
        return p * job["gpu_num"]


class CombinedEstimator:
    def __init__(self, args):
        if "Sept" in args.experiment_name:
            trace_range = ("2020-09-01 00:00:00", "2020-09-26 23:59:00")
        elif "July" in args.experiment_name:
            trace_range = ("2020-07-01 00:00:00", "2020-07-31 23:59:00")
        else:
            trace_range = None

        self.train_data, _ = utils.train_data_loader(args.trace_dir, trace_range)
        self.job_names = self.train_data["jobname"].unique()
        if args.scheduler == "lucid":
            if "Venus" in args.experiment_name:
                self.data = pd.read_csv(f"./estimator/ebm/{args.experiment_name}_ebm_weekly_updated.csv")
            else:
                raise NotImplementedError
            # NOTE
            self.data.loc[self.data["priority"] < 0, "priority"] = 1000
        else:
            # QSSF
            self.data = pd.read_csv(f"./estimator/lgb/{args.experiment_name}_lgb.csv")
        self.name = "CombinedEstimator"
        self.cluster_name = args.experiment_name.split("_")[0]

        """Hyperparameters"""
        self.alpha = 0.1  # priority = alpha * naive_inference + (1 - alpha) * lgbm_inference

        self.match_num = 2
        # Generate Random Priority for New User
        self.new_user_range = (100, 20000)
        self.agg_method = "mean"  # Mean or Median

    def update_train_data(self, job):
        job = pd.DataFrame(job)
        job = job.loc[:, "job_id":"duration"]
        # self.train_data = self.train_data.append(job, ignore_index=True)
        self.train_data = pd.concat([self.train_data, job], ignore_index=True)
        self.job_names = self.train_data["jobname"].unique()

    def check_history_job(self, job, user_data):
        gpu_data = user_data[user_data["gpu_num"] == job["gpu_num"]]

        # Simple Mean / Median
        if len(gpu_data) > 0:
            user_data = gpu_data
        return int(user_data["duration"].agg(self.agg_method) * job["gpu_num"])

        # # Weighted Windows  (Only recent 3 jobs)
        # if len(gpu_data) > 0:
        #     user_data = gpu_data
        # user_data = user_data.sort_values(by='submit_time')
        # if len(user_data) >= 3:
        #     user_data = user_data.iloc[-3:]
        #     job['random'] = user_data['job_id'].iloc[-1]
        # return int(user_data['duration'].agg(self.agg_method) * job['gpu_num'])

    def naive_inference(self, job):
        matches = difflib.get_close_matches(job["jobname"], self.job_names, n=self.match_num, cutoff=0.6)
        user_data = self.train_data[self.train_data["user"] == job["user"]]

        if len(matches) > 0:
            if matches[0] == job["jobname"]:
                user_data = user_data[user_data["jobname"] == job["jobname"]]
                if len(user_data) == 0:
                    return (
                        self.train_data[self.train_data["jobname"] == job["jobname"]]["duration"].agg(self.agg_method)
                        * job["gpu_num"]
                    )
                else:
                    return self.check_history_job(job, user_data)
            else:
                user_data = user_data[user_data["jobname"].isin(matches)]
                if len(user_data) == 0:
                    return (
                        self.train_data[self.train_data["jobname"].isin(matches)]["duration"].agg(self.agg_method)
                        * job["gpu_num"]
                    )
                else:
                    return self.check_history_job(job, user_data)
        else:
            job["random"] = 1  # estimate through randint/other job
            # return random.randint(100, 20000) * job['gpu_num']
            if len(user_data) == 0:  # new user
                return self.train_data["duration"].agg(self.agg_method) * job["gpu_num"]
            else:
                return user_data["duration"].agg(self.agg_method) * job["gpu_num"]

    def lgbm_inference(self, job):
        p = self.data[self.data["job_id"] == job["job_id"]]["priority"].iloc[0]
        return p * job["gpu_num"]

    def inference(self, job):
        lgbm = self.lgbm_inference(job)
        return lgbm

        # JobName are not provided
        # naive = self.naive_inference(job)
        # if lgbm > 0:
        #     priority = self.alpha * naive + (1 - self.alpha) * lgbm
        # else:
        #     priority = naive
        # return priority


class PhillyEstimator:
    def __init__(self, args):
        if args.scheduler == "lucid":
            self.data = pd.read_csv(f"./estimator/ebm/Philly_ebm.csv")
        else:
            self.data = pd.read_csv(f"./estimator/lgb/Philly_lgb.csv")

        self.args = args
        self.name = "PhillyEstimator"
        self.cluster_name = "Philly"

    def inference(self, job):
        p = self.data[self.data["job_id"] == job["job_id"]]["priority"].iloc[0]
        return p * job["gpu_num"]
