import argparse
import pandas as pd
import numpy as np
from datetime import timedelta


def set_interval(df, interval, agg):
    df_sampled = df.resample(interval).agg(agg)
    period = {"H": 24, "30min": 48, "10min": 144, "min": 1440}
    return df_sampled, period[interval]


"""Script for generating cluster sequence file"""


def parse_sequence(dir):
    date_range = ("2020-04-01 00:00:00", "2020-09-28 23:59:00")
    log = pd.read_csv(f"{dir}/cluster_log.csv", parse_dates=["submit_time", "start_time", "end_time"])
    cluster_gpu = pd.read_csv(f"{dir}/cluster_gpu_number.csv", parse_dates=["date"])

    df = pd.DataFrame(pd.date_range(start=date_range[0], end=date_range[1], freq="T"), columns=["time"])
    columns = [
        "running_gpujob_num",
        "total_gpu_num",
        "running_gpu_num",
        "running_gpu_multi",
        "running_gpu_single",
        "gpu_utilization",
        "gpu_utilization_multi",
        "gpu_utilization_single",
    ]
    df[columns] = 0

    log = log[log["gpu_num"] > 0]

    # parse each job
    for _, job in log.iterrows():
        gnum = job["gpu_num"]
        job["submit_time"] = job["submit_time"].replace(second=0)
        job["start_time"] = job["start_time"].replace(second=0)
        job["end_time"] = job["end_time"].replace(second=0)

        run = (df["time"] >= job["start_time"]) & (df["time"] <= job["end_time"])

        if gnum > 0:
            df.loc[run, "running_gpujob_num"] += 1
            df.loc[run, "running_gpu_num"] += gnum

            if gnum > 1:
                df.loc[run, "running_gpu_multi"] += gnum
            else:
                df.loc[run, "running_gpu_single"] += gnum

    cluster_gpu = cluster_gpu[["date", "total"]]
    cluster_gpu = cluster_gpu[
        (cluster_gpu["date"] >= pd.Timestamp(date_range[0])) & (cluster_gpu["date"] <= pd.Timestamp(date_range[1]))
    ]
    cluster_gpu = pd.concat([cluster_gpu] * 1440).sort_index()
    df["total_gpu_num"] = cluster_gpu["total"].values

    df["running_gpu_num"] = df["running_gpu_num"].combine(df["total_gpu_num"], min)
    df["gpu_utilization"] = (df["running_gpu_num"] / df["total_gpu_num"]).round(3)

    df["gpu_utilization_multi"] = (df["running_gpu_multi"] / df["total_gpu_num"]).round(3)
    df["gpu_utilization_single"] = (df["running_gpu_single"] / df["total_gpu_num"]).round(3)
    df.set_index("time", inplace=True, drop=True)
    return df


"""Script for generating cluster throughput file"""


def parse_throughput(dir):
    date_range = ("2020-04-01 00:00:00", "2020-09-28 23:50:00")
    log = pd.read_csv(f"{dir}/cluster_log.csv", parse_dates=["submit_time", "start_time", "end_time"])
    df = pd.DataFrame(pd.date_range(start=date_range[0], end=date_range[1], freq="10T"), columns=["time"],)
    df[
        [
            "submit_job_all",
            "start_job_all",
            "end_job_all",
            "submit_gpu_job",
            "start_gpu_job",
            "end_gpu_job",
            "submit_gpu_num",
            "start_gpu_num",
            "end_gpu_num",
        ]
    ] = 0
    df.set_index("time", inplace=True, drop=True)

    for i in range(len(df)):
        for kind in ("submit", "start", "end"):
            jobs = log[(log[kind + "_time"] >= df.index[i]) & (log[kind + "_time"] < df.index[i] + timedelta(minutes=10))]
            df[kind + "_job_all"][i] = len(jobs)
            df[kind + "_gpu_job"][i] = len(jobs[jobs["gpu_num"] != 0])
            df[kind + "_gpu_num"][i] = jobs["gpu_num"].agg(sum)
    return df


def parse_throughput_philly(dir):
    date_range = ("2017-10-01 00:00:00", "2017-11-18 23:50:00")
    log = pd.read_csv(f"{dir}/cluster_log.csv", parse_dates=["submit_time", "start_time", "end_time"])
    df = pd.DataFrame(pd.date_range(start=date_range[0], end=date_range[1], freq="10T"), columns=["time"],)
    df[["submit_gpu_job", "start_gpu_job", "end_gpu_job", "submit_gpu_num", "start_gpu_num", "end_gpu_num"]] = 0
    df.set_index("time", inplace=True, drop=True)

    for i in range(len(df)):
        for kind in ("submit", "start", "end"):
            jobs = log[(log[kind + "_time"] >= df.index[i]) & (log[kind + "_time"] < df.index[i] + timedelta(minutes=10))]
            # df.at[i, kind + "_gpu_job"] = len(jobs[jobs["gpu_num"] != 0])
            # df.at[i, kind + "_gpu_num"] = jobs["gpu_num"].agg(sum)
            df[kind + "_gpu_job"][i] = len(jobs[jobs["gpu_num"] != 0])
            df[kind + "_gpu_num"][i] = jobs["gpu_num"].agg(sum)
    return df


"""Script for generating cluster user file"""


def parse_user(dir, helios=False):
    # df   : contain cpu and gpu jobs
    # dfgpu: only gpu jobs
    # helios: analyze the whole data center users
    if helios:
        df = pd.read_csv(f"{dir}/all_cluster_log.csv", parse_dates=["submit_time", "start_time", "end_time"],)
    else:
        df = pd.read_csv(f"{dir}/cluster_log.csv", parse_dates=["submit_time", "start_time", "end_time"],)

    dfgpu = df[df["gpu_num"] > 0]

    users = df["user"].unique()
    users_gpu = dfgpu["user"].unique()

    user_df = pd.DataFrame({"user": users})
    user_df = user_df.set_index("user")
    user_df["cpu_only"] = False
    user_df[["vc_list", "vc_list_gpu"]] = None

    for u in users:
        data = df[df["user"] == u]
        datagpu = dfgpu[dfgpu["user"] == u]

        if u in users_gpu:
            cpu_job_num = len(data) - len(datagpu)

            plist = data["vc"].unique().tolist()
            user_df.at[u, "vc_list"] = plist
            user_df.at[u, "vc_num"] = len(plist)

            glist = datagpu["vc"].unique().tolist()
            user_df.at[u, "vc_list_gpu"] = glist
            user_df.at[u, "vc_num_gpu"] = len(glist)

            user_df.at[u, "job_num"] = len(data)
            user_df.at[u, "gpu_job_num"] = len(datagpu)
            user_df.at[u, "cpu_job_num"] = cpu_job_num

            user_df.at[u, "avg_run_time"] = data["duration"].mean()
            user_df.at[u, "avg_gpu_run_time"] = datagpu["duration"].mean()
            user_df.at[u, "avg_cpu_run_time"] = (
                0 if cpu_job_num == 0 else (data["duration"].sum() - datagpu["duration"].sum()) / cpu_job_num
            )

            user_df.at[u, "avg_pend_time"] = data["queue"].mean()
            user_df.at[u, "avg_gpu_pend_time"] = datagpu["queue"].mean()
            user_df.at[u, "avg_cpu_pend_time"] = (
                0 if cpu_job_num == 0 else (data["queue"].sum() - datagpu["queue"].sum()) / cpu_job_num
            )

            user_df.at[u, "avg_gpu_num"] = data["gpu_num"].mean()
            user_df.at[u, "avg_cpu_num"] = data["cpu_num"].mean()

            user_df.at[u, "total_gpu_time"] = (datagpu["gpu_num"] * datagpu["duration"]).sum()
            user_df.at[u, "total_cpu_time"] = (data["cpu_num"] * data["duration"]).sum()
            user_df.at[u, "total_cpu_only_time"] = (
                user_df.at[u, "total_cpu_time"] - (datagpu["cpu_num"] * datagpu["duration"]).sum()
            )
            user_df.at[u, "total_gpu_pend_time"] = datagpu["queue"].sum()

            user_df.at[u, "completed_percent"] = len(data[data["state"] == "COMPLETED"]) / len(data)
            user_df.at[u, "completed_gpu_percent"] = len(datagpu[datagpu["state"] == "COMPLETED"]) / len(datagpu)
            user_df.at[u, "completed_cpu_percent"] = (
                0
                if cpu_job_num == 0
                else (len(data[data["state"] == "COMPLETED"]) - len(datagpu[datagpu["state"] == "COMPLETED"])) / cpu_job_num
            )

            user_df.at[u, "cencelled_percent"] = len(data[data["state"] == "CANCELLED"]) / len(data)
            user_df.at[u, "cencelled_gpu_percent"] = len(datagpu[datagpu["state"] == "CANCELLED"]) / len(datagpu)
            user_df.at[u, "cencelled_cpu_percent"] = (
                0
                if cpu_job_num == 0
                else (len(data[data["state"] == "CANCELLED"]) - len(datagpu[datagpu["state"] == "CANCELLED"])) / cpu_job_num
            )

            user_df.at[u, "failed_percent"] = len(data[data["state"] == "FAILED"]) / len(data)
            user_df.at[u, "failed_gpu_percent"] = len(datagpu[datagpu["state"] == "FAILED"]) / len(datagpu)
            user_df.at[u, "failed_cpu_percent"] = (
                0
                if cpu_job_num == 0
                else (len(data[data["state"] == "FAILED"]) - len(datagpu[datagpu["state"] == "FAILED"])) / cpu_job_num
            )
        else:
            user_df.at[u, "cpu_only"] = True

            plist = data["vc"].unique().tolist()
            user_df.at[u, "vc_list"] = plist
            user_df.at[u, "vc_num"] = len(plist)

            user_df.at[u, "vc_list_gpu"] = []
            user_df.at[u, "vc_num_gpu"] = 0

            user_df.at[u, "job_num"] = len(data)
            user_df.at[u, "gpu_job_num"] = 0
            user_df.at[u, "cpu_job_num"] = len(data)

            user_df.at[u, "avg_run_time"] = data["duration"].mean()
            user_df.at[u, "avg_gpu_run_time"] = 0
            user_df.at[u, "avg_cpu_run_time"] = user_df.at[u, "avg_run_time"]

            user_df.at[u, "avg_pend_time"] = data["queue"].mean()
            user_df.at[u, "avg_gpu_pend_time"] = 0
            user_df.at[u, "avg_cpu_pend_time"] = user_df.at[u, "avg_pend_time"]

            user_df.at[u, "avg_gpu_num"] = 0
            user_df.at[u, "avg_cpu_num"] = data["cpu_num"].mean()

            user_df.at[u, "total_gpu_time"] = 0
            user_df.at[u, "total_cpu_time"] = (data["cpu_num"] * data["duration"]).sum()
            user_df.at[u, "total_cpu_only_time"] = user_df.at[u, "total_cpu_time"]
            user_df.at[u, "total_gpu_pend_time"] = 0

            user_df.at[u, "completed_percent"] = len(data[data["state"] == "COMPLETED"]) / len(data)
            user_df.at[u, "completed_gpu_percent"] = 0
            user_df.at[u, "completed_cpu_percent"] = user_df.at[u, "completed_percent"]

            user_df.at[u, "cencelled_percent"] = len(data[data["state"] == "CANCELLED"]) / len(data)
            user_df.at[u, "cencelled_gpu_percent"] = 0
            user_df.at[u, "cencelled_cpu_percent"] = user_df.at[u, "cencelled_percent"]

            user_df.at[u, "failed_percent"] = len(data[data["state"] == "FAILED"]) / len(data)
            user_df.at[u, "failed_gpu_percent"] = 0
            user_df.at[u, "failed_cpu_percent"] = user_df.at[u, "failed_percent"]

    user_df.sort_values(by="job_num", ascending=False, inplace=True)
    user_df[["vc_num", "vc_num_gpu", "job_num", "gpu_job_num", "cpu_job_num"]] = user_df[
        ["vc_num", "vc_num_gpu", "job_num", "gpu_job_num", "cpu_job_num"]
    ].astype(int)
    return user_df


"""Script for generating a all cluster trace file"""


def parse_all_cluster_log(clusters):
    logall = pd.DataFrame()

    for cluster in clusters:
        print(f"Parsing {cluster}")
        data_dir = f"../data/{cluster}"
        log = pd.read_csv(f"{data_dir}/cluster_log.csv", parse_dates=["submit_time", "start_time", "end_time"])
        log["c"] = cluster
        logall = pd.concat([logall, log])

    logall.insert(1, "cluster", logall["c"])
    logall.drop(columns="c", inplace=True)
    logall.to_csv("../data/all_cluster_log.csv", index=False)


"""Script for generating all cluster monthly file"""


def parse_monthly_job(cluster_name):
    df = pd.DataFrame(cluster_name, columns=["id"])
    df.set_index("id", drop=True, inplace=True)

    for cluster in cluster_name:
        data_dir = f"../data/{cluster}"
        log = pd.read_csv(f"{data_dir}/cluster_log.csv", parse_dates=["submit_time", "start_time", "end_time"])
        log["month"] = getattr(log["submit_time"].dt, "month").astype(np.int16)
        glog = log[log["gpu_num"] > 0]
        clog = log[log["gpu_num"] == 0]
        for m in range(4, 10):
            month_glog = glog[glog["month"] == m]
            df.at[cluster, str(m) + " Job Num"] = len(log[log["month"] == m])
            df.at[cluster, str(m) + " CPU Job Num"] = len(clog[clog["month"] == m])
            df.at[cluster, str(m) + " GPU Job Num"] = len(month_glog)
            df.at[cluster, str(m) + " GJobNum 1"] = len(month_glog[month_glog["gpu_num"] == 1])
            df.at[cluster, str(m) + " GJobNum g1"] = len(month_glog[month_glog["gpu_num"] > 1])

    df = df.astype(int)
    df.to_csv("../data/all_cluster_monthly_job_num.csv")


def parse_monthly_util(cluster_list):
    df = pd.DataFrame()

    for i in range(len(cluster_list)):
        seq = pd.read_csv(f"../data/{cluster_list[i]}/cluster_sequence.csv", parse_dates=["time"], index_col="time",)
        seq["month"] = seq.index.month
        df.loc[:, cluster_list[i]] = seq.groupby("month").mean()["gpu_utilization"].values

    df.to_csv("../data/all_cluster_monthly_utilization.csv", index=False)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Trace Parser")
    # parser.add_argument("-c", "--cluster-list", default=["Venus", "Earth", "Saturn", "Uranus"], help="Cluster list for parsing")
    # args = parser.parse_args()

    # clusters = args.cluster_list
    # if type(clusters) is not list:
    #     clusters = [clusters]

    # print("1/4. Generating all cluster log file.")
    # parse_all_cluster_log(clusters)

    # print("2/4. Generating cluster sequence and throughput files. This step may take hours.")
    # for cluster in clusters:
    #     print(f"Parsing {cluster}")
    #     data_dir = f"../data/{cluster}"
    #     # seq = parse_sequence(data_dir)
    #     df = parse_throughput(data_dir)

    #     # seq.to_csv(f"{data_dir}/cluster_sequence.csv")
    #     df.to_csv(f"{data_dir}/cluster_throughput.csv")

    # print("3/4. Generating monthly job num and utilization files")
    # parse_monthly_job(clusters)
    # parse_monthly_util(clusters)

    # print("4/4. Generating monthly job num and utilization files")
    # data_dir = f"../data"
    # df = parse_user(data_dir, helios=True)
    # df.to_pickle(f"{data_dir}/cluster_user.pkl")

    # for cluster in clusters:
    #     print(f"Parsing {cluster}")
    #     data_dir = f"../data/{cluster}"
    #     df = parse_user(data_dir)

    #     df.to_pickle(f"{data_dir}/cluster_user.pkl")

    print("Philly Throughput. This step may take hours.")
    data_dir = f"./Philly"
    # seq = parse_sequence(data_dir)
    df = parse_throughput_philly(data_dir)

    # seq.to_csv(f"{data_dir}/cluster_sequence.csv")
    df.to_csv(f"{data_dir}/cluster_throughput.csv")
