import argparse
import pandas as pd
from pathlib import Path
import datetime
import pickle

"""
Generate VC configuration for simulation
"""


def main(args):
    cluster = args.cluster
    date = args.date

    if not Path(f"./{cluster}").exists():
        Path(f"./{cluster}").mkdir()

    if cluster == "Philly":
        df = pd.read_csv(Path(f"./philly_vc.csv"))
    else:
        file = Path(f"../../../data/{cluster}/cluster_gpu_number.csv")
        df = pd.read_csv(file, parse_dates=["date"])

    if cluster == "Jupiter" and date == "July":
        vcs = df.columns.values[1:-2]
        nodes_num = df[df["date"] == datetime.datetime(2020, 7, 1)].values[0][1:-2]
        nodes_num = nodes_num // 8

        assert len(nodes_num) == len(vcs)
        dic = dict(zip(vcs, nodes_num))
        del dic["vc7Bz"]  # Not Consider Test VC

        with open(f"./{cluster}/vc_dict.pkl", "wb") as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    if cluster == "Neptune" and date == "Sept":
        vcs = df.columns.values[1:-1]
        nodes_num = df[df["date"] == datetime.datetime(2020, 9, 1)].values[0][1:-1]
        nodes_num = nodes_num // 8

        assert len(nodes_num) == len(vcs)
        dic = dict(zip(vcs, nodes_num))

        del dic["vc7Bz"]  # Not Consider Test VC
        del dic["vcIoD"]  # Zero nodes
        del dic["vcftk"]  # Zero nodes
        del dic["vc5LC"]  # No Job
        del dic["vcEwI"]  # No Job

        # Some vc rename / merge together
        del dic["vcrsE"]  # Merge to vcBUL
        del dic["vcHyk"]  # Merge to vcBUL
        del dic["vcVvI"]  # Merge to vcUV3

        with open(f"./{cluster}/vc_dict.pkl", "wb") as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    if cluster == "Saturn" and date == "Sept":
        vcs = df.columns.values[1:-1]
        nodes_num = df[df["date"] == datetime.datetime(2020, 9, 1)].values[0][1:-1]
        nodes_num = nodes_num // 8

        assert len(nodes_num) == len(vcs)
        dic = dict(zip(vcs, nodes_num))

        del dic["vc7Bz"]  # Not Consider Test VC
        del dic["vcHcQ"]  # Zero nodes
        del dic["vck1d"]  # Zero nodes
        del dic["vcj72"]  # Zero nodes
        del dic["vcIya"]  # Zero nodes
        del dic["vcygX"]  # Zero nodes
        del dic["vcxqr"]  # Zero nodes
        del dic["vcsgw"]  # Zero nodes

        with open(f"./{cluster}/vc_dict.pkl", "wb") as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    if cluster == "Uranus" and date == "Sept":
        vcs = df.columns.values[1:-1]
        nodes_num = df[df["date"] == datetime.datetime(2020, 9, 10)].values[0][1:-1]
        nodes_num = nodes_num // 8

        assert len(nodes_num) == len(vcs)
        dic = dict(zip(vcs, nodes_num))

        del dic["vc7Bz"]  # Not Consider Test VC
        del dic["vczGr"]  # Zero nodes
        del dic["vciN1"]  # Zero nodes
        del dic["vcV7h"]  # Zero nodes
        del dic["vcRAl"]  # Zero nodes
        del dic["vcvcM"]  # Zero nodes
        del dic["vc1z2"]  # Zero nodes

        del dic["vcHyk"]  # Merge to vcOlr
        del dic["vcRDh"]  # Merge to vc7hD
        del dic["vcFsC"]  # Merge to vc7hD
        del dic["vcVvI"]  # Merge to vcUV3
        del dic["vcxqr"]  # Merge to vcUV3
        del dic["vcsBT"]  # Merge to vcUV3
        del dic["vcygX"]  # Merge to vcUV3

        with open(f"./{cluster}/vc_dict.pkl", "wb") as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    if cluster == "Earth" and date == "Sept":
        vcs = df.columns.values[1:-1]
        nodes_num = df[df["date"] == datetime.datetime(2020, 9, 10)].values[0][1:-1]
        nodes_num = nodes_num // 8

        assert len(nodes_num) == len(vcs)
        dic = dict(zip(vcs, nodes_num))

        # del dic['vc7Bz']  # Not Consider Test VC
        del dic["vcp4O"]  # Zero nodes
        del dic["vcvcM"]  # Zero nodes
        del dic["vcXrB"]  # Zero nodes
        del dic["vc7hD"]  # Zero nodes
        del dic["vcIya"]  # Zero nodes
        del dic["vc8Sj"]  # Zero nodes

        del dic["vcLJZ"]  # Zero nodes in Sept

        del dic["vcxS0"]  # Merge to vc3sl

        with open(f"./{cluster}/vc_dict.pkl", "wb") as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    if cluster == "Venus" and date == "Sept":
        vcs = df.columns.values[1:-1]
        nodes_num = df[df["date"] == datetime.datetime(2020, 9, 10)].values[0][1:-1]
        nodes_num = nodes_num // 8

        assert len(nodes_num) == len(vcs)
        dic = dict(zip(vcs, nodes_num))

        # del dic['vc7Bz']  # Not Consider Test VC
        del dic["vcEhP"]  # Zero nodes
        del dic["vcIya"]  # Zero nodes
        del dic["vcJLV"]  # Zero nodes
        del dic["vcJkd"]  # Zero nodes
        del dic["vcsBT"]  # Zero nodes

        del dic["vcbIW"]  # Merge to vcvGl
        del dic["vc6YE"]  # Merge to vcvGl
        del dic["vcOhe"]  # Merge to vcKeu
        del dic["vccJW"]  # Merge to vcKeu
        del dic["vcP2J"]  # Merge to vchA3

        with open(f"./{cluster}/vc_dict.pkl", "wb") as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    if cluster == "Philly":
        d = dict(zip(df["vc"].values, df["node num"].values))

        with open(f"./Philly/vc_dict_homo.pkl", "wb") as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VC Configuration Generator")
    parser.add_argument("-c", "--cluster", default="Earth", type=str, help="Cluster Name")
    parser.add_argument("-d", "--date", default="Sept", type=str, help="Month")
    args = parser.parse_args()
    main(args)
