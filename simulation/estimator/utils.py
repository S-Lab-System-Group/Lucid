import pandas as pd
import sys
import datetime
import logging


def train_data_loader(dir, test_date_range):
    # NOTE: test_date_range
    # test_date_range refer to the dates used for test, which will be excluded in our train dataset
    start = "2020-04-01 00:00:00"
    df = pd.read_csv(
        dir + "/cluster_full_log.csv",
        parse_dates=["submit_time"],
        usecols=["job_id", "user", "vc", "jobname", "gpu_num", "cpu_num", "submit_time", "duration"],
    )

    # Consider gpu jobs only
    df = df[df["gpu_num"] > 0]
    df = df.sort_values(by="submit_time")

    # VC filter
    vc_df = pd.read_csv(dir + "/vc_config.csv", index_col=0)
    vc_list = vc_df.index.to_list()
    df = df[df["vc"].isin(vc_list)]

    df = df[df["submit_time"] >= pd.Timestamp(start)]
    df["submit_time"] = df["submit_time"].apply(lambda x: int(datetime.datetime.timestamp(pd.Timestamp(x))))

    # Normalizing
    df["submit_time"] = df["submit_time"] - df.iloc[0]["submit_time"]

    # Slicing val data
    begin = (pd.Timestamp(test_date_range[0]) - pd.Timestamp(start)).total_seconds()
    end = (pd.Timestamp(test_date_range[1]) - pd.Timestamp(start)).total_seconds()
    val_df = df[(df["submit_time"] >= begin) & (df["submit_time"] <= end)]
    # Slicing train data
    # | (df['submit_time'] > pd.Timestamp(test_date_range[1]))]
    train_df = df[(df["submit_time"] < begin)]

    # Filter user, vc not in val data  around 9% jobs be filtered
    val_users = val_df["user"].unique()

    val_vcs = val_df["vc"].unique()

    train_df = train_df[train_df["user"].isin(val_users)]
    train_df = train_df[train_df["vc"].isin(val_vcs)]  # no jobs be filtered

    train_df = train_df.sort_values(by="submit_time")
    train_df.reset_index(inplace=True, drop=True)

    val_df = val_df.sort_values(by="submit_time")
    val_df.reset_index(inplace=True, drop=True)

    return train_df, val_df


def logger_init(file):
    logger = logging.getLogger()
    handler_file = logging.FileHandler(f"{file}.log", "w")
    handler_stream = logging.StreamHandler(sys.stdout)

    logger.setLevel(logging.INFO)
    handler_file.setLevel(logging.INFO)
    handler_stream.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(processName)s | %(message)s", datefmt="%Y %b %d %H:%M:%S")
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)

    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)

    return logger
