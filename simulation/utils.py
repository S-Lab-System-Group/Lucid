import profile
import sys
import os
import logging
import datetime
import pandas as pd
from job import Job, Trace
from policy import (
    ShortestJobFirst,
    FirstInFirstOut,
    ShortestRemainingTimeFirst,
    QuasiShortestServiceFirst,
    Lucid,
    Tiresias,
)
from profiler import LeastGPUFirstProfiler

sys.path.append("..")

PROFILER_ENABLED_SCHEDULERS = ["lucid"]


def simulate_vc(trace, vc, placement, log_dir, policy, logger, start_ts, *args):
    if policy == "sjf":
        scheduler = ShortestJobFirst(trace, vc, placement, log_dir, logger, start_ts)
    elif policy == "fifo":
        scheduler = FirstInFirstOut(trace, vc, placement, log_dir, logger, start_ts)
    elif policy == "srtf":
        scheduler = ShortestRemainingTimeFirst(trace, vc, placement, log_dir, logger, start_ts)
    elif policy == "qssf":
        scheduler = QuasiShortestServiceFirst(trace, vc, placement, log_dir, logger, start_ts, args[0])
    elif policy == "lucid":
        scheduler = Lucid(trace, vc, placement, log_dir, logger, start_ts, args[0], args[1])
    elif policy == "tiresias":
        scheduler = Tiresias(trace, vc, placement, log_dir, logger, start_ts)
    scheduler.simulate()
    logger.info(f"Finish {vc.vc_name}")
    return True


def trace_profile(trace, scale, time_limit, profiler_factor, placement, log_dir, logger, start_ts):
    profiler = LeastGPUFirstProfiler(trace, scale, time_limit, profiler_factor, placement, log_dir, logger, start_ts)
    profiler.profile()
    trace.reset_trace()
    logger.info("Finish Profiling")
    return trace


def get_available_schedulers():
    return ["fifo", "sjf", "srtf", "qssf", "lucid", "tiresias"]


def get_sweep_schedulers():
    return ["fifo", "sjf", "srtf", "qssf", "tiresias"]


def get_available_placers():
    return ["random", "consolidate", "consolidateFirst"]


def trace_process(dir, date_range, read_full):
    start = "2020-04-01 00:00:00"
    if read_full == False:
        df = pd.read_csv(
            dir + "/cluster_log.csv",
            parse_dates=["submit_time"],
            usecols=["job_id", "user", "vc", "jobname", "gpu_num", "cpu_num", "state", "submit_time", "duration"],
        )
    else:
        df = pd.read_csv(
            dir + "/cluster_full_log.csv",
            parse_dates=["submit_time"],
            usecols=[
                "job_id",
                "user",
                "vc",
                "jobname",
                "gpu_num",
                "cpu_num",
                "state",
                "submit_time",
                "duration",
                "dataset",
                "model",
                "batchsize",
                "amp",
                "speed",
                "gpu_util",
                "gmem_util",
                "gmem",
            ],
        )
    # Consider gpu jobs only
    df = df[df["gpu_num"] > 0]

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df.sort_values(by="submit_time", inplace=True)
    df = df[df["submit_time"] >= pd.Timestamp(start)]
    df["submit_time"] = df["submit_time"].apply(lambda x: int(datetime.datetime.timestamp(pd.Timestamp(x))))

    # Normalizing
    df["submit_time"] = df["submit_time"] - df.iloc[0]["submit_time"]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None

    # Slicing simulation part
    begin = (pd.Timestamp(date_range[0]) - pd.Timestamp(start)).total_seconds()
    end = (pd.Timestamp(date_range[1]) - pd.Timestamp(start)).total_seconds()
    df = df[(df["submit_time"] >= begin) & (df["submit_time"] <= end)]

    df.sort_values(by="submit_time", inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df, begin


def trace_real_process(dir):
    df = pd.read_csv(
        dir + "/cluster_full_log.csv",
        parse_dates=["submit_time"],
        usecols=[
            "job_id",
            "user",
            "vc",
            "jobname",
            "gpu_num",
            "cpu_num",
            "state",
            "submit_time",
            "duration",
            "dataset",
            "model",
            "batchsize",
            "amp",
            "speed",
            "gpu_util",
            "gmem_util",
            "gmem",
        ],
    )

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None
    df["submit_time"] = df["submit_time"].astype(float)
    df["submit_time"] = df["submit_time"].astype(int)
    df.reset_index(inplace=True, drop=True)

    return df, 0


def trace_pollux_process(dir, idx):
    df = pd.read_csv(
        f"{dir}/cluster_full_log_{idx}.csv",
        parse_dates=["submit_time"],
        usecols=[
            "job_id",
            "user",
            "vc",
            "jobname",
            "gpu_num",
            "cpu_num",
            "state",
            "submit_time",
            "duration",
            "dataset",
            "model",
            "batchsize",
            "amp",
            "speed",
            "gpu_util",
            "gmem_util",
            "gmem",
        ],
    )

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None
    df["submit_time"] = df["submit_time"].astype(float)
    df["submit_time"] = df["submit_time"].astype(int)
    df.reset_index(inplace=True, drop=True)

    return df, 0


def trace_philly_process(dir, date_range, read_full):
    start = "2017-10-01 00:00:00"
    if read_full == False:
        df = pd.read_csv(
            dir + "/cluster_log.csv",
            parse_dates=["submit_time"],
            usecols=["user", "vc", "jobname", "gpu_num", "state", "submit_time", "duration"],
        )
    else:
        df = pd.read_csv(
            dir + "/cluster_full_log.csv",
            parse_dates=["submit_time"],
            usecols=[
                "user",
                "vc",
                "job_id",
                "gpu_num",
                "state",
                "submit_time",
                "duration",
                "dataset",
                "model",
                "batchsize",
                "amp",
                "speed",
                "gpu_util",
                "gmem_util",
                "gmem",
            ],
        )
    # Consider gpu jobs only
    df = df[df["gpu_num"] > 0]
    df.sort_values(by="submit_time", inplace=True)
    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df = df[df["submit_time"] >= pd.Timestamp(start)]
    df["submit_time"] = df["submit_time"].apply(lambda x: int(datetime.datetime.timestamp(pd.Timestamp(x))))

    df.rename(columns={"jobname": "job_id"}, inplace=True)
    df["state"] = df["state"].replace("Pass", "COMPLETED")
    df["state"] = df["state"].replace("Failed", "FAILED")
    df["state"] = df["state"].replace("Killed", "CANCELLED")

    # Normalizing
    df["submit_time"] = df["submit_time"] - df.iloc[0]["submit_time"]

    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "queue", "jct"]] = 0
    df["status"] = None

    # Slicing simulation part
    begin = (pd.Timestamp(date_range[0]) - pd.Timestamp(start)).total_seconds()
    end = (pd.Timestamp(date_range[1]) - pd.Timestamp(start)).total_seconds()
    df = df[(df["submit_time"] >= begin) & (df["submit_time"] <= end)]

    df.sort_values(by="submit_time", inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df, begin


def trace_parser(df):
    trace = Trace()

    for _, series in df.iterrows():
        trace.append_job(Job(series))
    trace.sort_jobs("submit_time")
    return trace


def logger_init(file):
    logger = logging.getLogger()
    handler_file = logging.FileHandler(f"{file}.log", "w")
    handler_stream = logging.StreamHandler()  # sys.stdout

    logger.setLevel(logging.INFO)
    handler_file.setLevel(logging.INFO)
    handler_stream.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(processName)s | %(message)s", datefmt="%Y %b %d %H:%M:%S")
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)

    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)

    return logger


def cluster_concatenate(policy, placer, log_dir, dir):
    prefix = f"{policy}_{placer}"
    if not os.path.exists(log_dir + "/all"):
        os.mkdir(log_dir + "/all")

    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vcs = vc_df.index.to_list()

    """Log"""
    cluster_log = pd.DataFrame()
    for vc in vcs:
        vc_log = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_log.csv")
        cluster_log = pd.concat([cluster_log, vc_log])
    cluster_log.sort_values(by="submit_time", inplace=True)
    cluster_log.to_csv(f"{log_dir}/all/{prefix}_all_log.csv", index=False)

    """Seq"""
    cluster_seq = pd.DataFrame()
    add_list = [
        "total_gpu_num",
        "idle_gpu_num",
        "pending_gpu_num",
        "running_gpujob_num",
        "pending_gpujob_num",
        "pending_job_num_less_8",
        "total_node_num",
        "consolidate_node_num",
        "shared_node_num",
    ]
    for vc in vcs:
        vc_seq = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_seq.csv")
        if len(cluster_seq) == 0:
            cluster_seq = vc_seq
            continue
        cluster_seq[add_list] = cluster_seq[add_list] + vc_seq[add_list]
        cluster_seq.dropna(inplace=True)
        cluster_seq = cluster_seq.astype(int)
        cluster_seq["gpu_utilization"] = (
            (cluster_seq["total_gpu_num"] - cluster_seq["idle_gpu_num"]) / cluster_seq["total_gpu_num"]
        ).round(3)
    cluster_seq.to_csv(f"{log_dir}/all/{prefix}_all_seq.csv", index=False)


def cluster_analysis(placer, log_dir, dir):
    """Generate Algorithm Comparsion CSV"""
    # ignore_warm_up = start_ts + 7*24*3600

    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vcs = vc_df.index.to_list()
    vcs.append("all")

    files = os.listdir(f"{log_dir}/all")
    prefix = set()
    for file in files:
        policy = file.split("_")[0]
        placer = file.split("_")[1]
        prefix.add(f"{policy}_{placer}")
    prefix_list = sorted(list(prefix))

    # prefix_list = []
    # for i in get_available_schedulers():
    #     prefix = f"{i}_{placer}"
    #     prefix_list.append(prefix)

    jct_avg = pd.DataFrame()
    que_avg = pd.DataFrame()
    for prefix in prefix_list:
        for vc in vcs:
            vc_log = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_log.csv")
            # vc_log = vc_log[vc_log['submit_time'] > ignore_warm_up]
            jct_avg.at[vc, prefix] = vc_log["jct"].mean()
            que_avg.at[vc, prefix] = vc_log["queue"].mean()

    jct_avg = jct_avg.astype(int)
    que_avg = que_avg.astype(int)
    jct_avg.to_csv(f"{log_dir}/jct_avg_{placer}.csv")
    que_avg.to_csv(f"{log_dir}/que_avg_{placer}.csv")


def get_trace(experiment_name, trace_dir, read_full, idx=None):
    if "Philly" in experiment_name:
        trace_range = ("2017-10-01 00:00:00", "2017-10-07 23:59:00")
        trace_df, start_ts = trace_philly_process(trace_dir, trace_range, read_full)
    elif "Pollux" in experiment_name:
        trace_df, start_ts = trace_pollux_process(trace_dir, idx)
    else:
        if "Sept" in experiment_name:
            trace_range = ("2020-09-01 00:00:00", "2020-09-26 23:59:00")
            trace_df, start_ts = trace_process(trace_dir, trace_range, read_full)
        elif "July" in experiment_name:
            trace_range = ("2020-07-01 00:00:00", "2020-07-31 23:59:00")
            trace_df, start_ts = trace_process(trace_dir, trace_range, read_full)
        else:
            raise ValueError

    return trace_df, start_ts


def profiler_config(experiment_name, vc_dict):
    cluster = experiment_name.split("_")[0]
    profile_scale = {"Venus": 2, "Philly": 2}
    profile_time = {"Venus": 200, "Philly": 80}
    profile_factor = {"Venus": 4, "Philly": 2}

    # Basic Config
    scale, time, factor = profile_scale[cluster], profile_time[cluster], profile_factor[cluster]
    if cluster == "Philly":
        vc_dict["philly"] -= scale
    elif cluster == "Venus":
        vc_dict["vc8Gr"] -= 1
        vc_dict["vcefl"] -= 1
        # vc_dict["vcYVn"] -= 1  # For elastic scaling
    return vc_dict, scale, time, factor


def check_profiler_scale_available(experiment_name, scale, vc_dict, prof_locate_vc=None):
    # Use only for debug
    default_vc = {
        "Venus": "vc8Gr",
        "Saturn": "vcqdr",
        "Philly": "philly",
    }
    cluster = experiment_name.split("_")[0]

    if not prof_locate_vc:
        vc = default_vc[cluster]

    if scale <= vc_dict[vc]:
        return vc
    else:
        raise ValueError("Profile Node Scale Exceed VC Capacity")


if __name__ == "__main__":
    files = os.listdir(f"log/Venus_Sept/all")
    prefix = set()
    for file in files:
        policy = file.split("_")[0]
        placer = file.split("_")[1]
        prefix.add(f"{policy}_{placer}")
    prefix_list = sorted(list(prefix))
    print(prefix_list)
