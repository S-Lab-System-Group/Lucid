import argparse
import pandas as pd
from pathlib import Path

"""
Processing log files for simulation
"""


def main(args):
    cluster = args.cluster

    if not Path(f"./{cluster}").exists():
        Path(f"./{cluster}").mkdir()

    if cluster == "Philly":
        logfile = Path(f"../../../analysis/1_compare with Philly trace/philly_trace.csv")
    else:
        logfile = Path(f"../../../data/{cluster}/cluster_log.csv")

    df = pd.read_csv(logfile, parse_dates=["submit_time", "start_time", "end_time"])
    df = df.sort_values(by="submit_time")
    df.reset_index(drop=True, inplace=True)

    if cluster == "Neptune":
        df = df[df["vc"] != "vc7Bz"]
        df = df[df["vc"] != "vcIoD"]
        df = df[df["vc"] != "vcftk"]
        df = df[df["vc"] != "vc5LC"]
        df = df[df["vc"] != "vcEwI"]

        df.loc[df["vc"] == "vcVvI", "vc"] = "vcUV3"
        df.loc[df["vc"] == "vcrsE", "vc"] = "vcBUL"
        df.loc[df["vc"] == "vcHyk", "vc"] = "vcBUL"

        df.reset_index(drop=True, inplace=True)
        df.to_csv(f"./{cluster}/cluster_log.csv", index=False)

    elif cluster == "Saturn":
        df = df[df["vc"] != "vc7Bz"]
        df = df[df["vc"] != "vcHcQ"]
        df = df[df["vc"] != "vck1d"]
        df = df[df["vc"] != "vcj72"]
        df = df[df["vc"] != "vcIya"]
        df = df[df["vc"] != "vcygX"]
        df = df[df["vc"] != "vcxqr"]
        df = df[df["vc"] != "vcsgw"]

        df.reset_index(drop=True, inplace=True)
        df.to_csv(f"./{cluster}/cluster_log.csv", index=False)

    elif cluster == "Uranus":
        df = df[df["vc"] != "vc7Bz"]
        df = df[df["vc"] != "vczGr"]
        df = df[df["vc"] != "vciN1"]
        df = df[df["vc"] != "vcV7h"]
        df = df[df["vc"] != "vcRAl"]
        df = df[df["vc"] != "vcvcM"]
        df = df[df["vc"] != "vc1z2"]

        df.loc[df["vc"] == "vcVvI", "vc"] = "vcUV3"
        df.loc[df["vc"] == "vcxqr", "vc"] = "vcUV3"
        df.loc[df["vc"] == "vcsBT", "vc"] = "vcUV3"
        df.loc[df["vc"] == "vcygX", "vc"] = "vcUV3"
        df.loc[df["vc"] == "vcHyk", "vc"] = "vcOlr"
        df.loc[df["vc"] == "vcRDh", "vc"] = "vc7hD"
        df.loc[df["vc"] == "vcFsC", "vc"] = "vc7hD"

        df.reset_index(drop=True, inplace=True)
        df.to_csv(f"./{cluster}/cluster_log.csv", index=False)

    elif cluster == "Earth":
        df = df[df["vc"] != "vcp4O"]
        df = df[df["vc"] != "vcvcM"]
        df = df[df["vc"] != "vcXrB"]
        df = df[df["vc"] != "vc7hD"]
        df = df[df["vc"] != "vcIya"]
        df = df[df["vc"] != "vc8Sj"]
        df = df[df["vc"] != "vcLJZ"]

        df.loc[df["vc"] == "vcxS0", "vc"] = "vc3sl"

        df.reset_index(drop=True, inplace=True)
        df.to_csv(f"./{cluster}/cluster_log.csv", index=False)

    elif cluster == "Venus":
        df = df[df["vc"] != "vcEhP"]
        df = df[df["vc"] != "vcIya"]
        df = df[df["vc"] != "vcJLV"]
        df = df[df["vc"] != "vcJkd"]
        df = df[df["vc"] != "vcsBT"]

        df.loc[df["vc"] == "vcbIW", "vc"] = "vcvGl"
        df.loc[df["vc"] == "vc6YE", "vc"] = "vcvGl"
        df.loc[df["vc"] == "vcOhe", "vc"] = "vcKeu"
        df.loc[df["vc"] == "vccJW", "vc"] = "vcKeu"
        df.loc[df["vc"] == "vcP2J", "vc"] = "vchA3"

        df.reset_index(drop=True, inplace=True)
        df.to_csv(f"./{cluster}/cluster_log.csv", index=False)

    elif cluster == "Philly":
        df = df[df["vc"] != "795a4c"]
        df = df[df["vc"] != "51b7ef"]
        df = df[df["vc"] != "925e2b"]
        df = df[df["vc"] != "23dbec"]

        df.reset_index(drop=True, inplace=True)
        df.to_csv(f"./{cluster}/cluster_log.csv", index=False)

    else:
        raise ValueError("Wrong Cluster Name.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Log Processor")
    parser.add_argument("-c", "--cluster", default="Earth", type=str, help="Cluster Name")
    args = parser.parse_args()
    main(args)
