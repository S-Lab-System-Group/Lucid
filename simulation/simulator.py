import time
import os
import argparse
import multiprocessing

import pandas as pd
import utils
import cluster
from estimator import CombinedEstimator, PhillyEstimator
from updater import ColocateUpdater

os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())


def main(args):
    code_start = time.perf_counter()

    """Logger Setting"""
    log_dir = f"{args.log_dir}/{args.experiment_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir + "/logfile")
    logger = utils.logger_init(file=f"{log_dir}/logfile/{args.scheduler}_{args.placer}")

    """Infrastructure & Trace Initialization"""
    vc_df = pd.read_csv(args.trace_dir + "/vc_config.csv", index_col=0)
    vc_dict = vc_df.to_dict()["num"]

    trace_df, start_ts = utils.get_trace(args.experiment_name, args.trace_dir, read_full=True, idx=args.pollux_idx)

    logger.info(f"Total Job Number in Cluster Training: {len(trace_df)}")

    trace = utils.trace_parser(trace_df)

    if args.scheduler in utils.PROFILER_ENABLED_SCHEDULERS and not args.sweep:
        if args.profiler_auto:
            vc_dict, prof_scale, prof_time, prof_factor = utils.profiler_config(args.experiment_name, vc_dict)
            trace = utils.trace_profile(trace, prof_scale, prof_time, prof_factor, args.placer, log_dir, logger, start_ts)
        else:
            # NOTE: NOT update vc_dict for manual configuration
            prof_vc = utils.check_profiler_scale_available(
                args.experiment_name, args.profiler_scale, vc_dict, prof_locate_vc=None
            )
            trace = utils.trace_profile(
                trace, args.profiler_scale, args.profiler_time, args.profiler_factor, args.placer, log_dir, logger, start_ts
            )
            return
        logger.info(f"Profiling Execution Time: {round(time.perf_counter() - code_start, 2)}s")

    colocate_df = pd.read_csv("data/colocate_info.csv")
    updater = ColocateUpdater(colocate_df)

    CLUSTER = cluster.Cluster(vc_dict, args.num_gpus_per_node, args.num_cpus_per_node)

    if "Philly" in args.experiment_name:
        estimator = PhillyEstimator(args)
    else:
        estimator = CombinedEstimator(args)

    """
    Sweep ON: Run All Scheduler Policies in One Experiment
    Sweep OFF: Run Dedicated Scheduler Policy (Default)
    """
    if args.sweep:
        process_num = os.cpu_count()
        all_args_list = []
        for policy in utils.get_sweep_schedulers():
            if policy == "qssf":
                for i in range(len(vc_dict)):
                    all_args_list.append((trace, CLUSTER.vc_list[i], args.placer, log_dir, policy, logger, start_ts, estimator))
            elif policy == "lucid":
                for i in range(len(vc_dict)):
                    all_args_list.append(
                        (trace, CLUSTER.vc_list[i], args.placer, log_dir, policy, logger, start_ts, estimator, updater)
                    )
            else:
                for i in range(len(vc_dict)):
                    all_args_list.append((trace, CLUSTER.vc_list[i], args.placer, log_dir, policy, logger, start_ts))
    else:
        if args.processes is None:
            process_num = min(len(CLUSTER.vc_list), os.cpu_count())
        else:
            process_num = args.processes

        all_args_list = []
        for i in range(len(vc_dict)):
            if args.scheduler == "qssf":
                all_args_list.append(
                    (trace, CLUSTER.vc_list[i], args.placer, log_dir, args.scheduler, logger, start_ts, estimator)
                )
            elif args.scheduler == "lucid":
                all_args_list.append(
                    (trace, CLUSTER.vc_list[i], args.placer, log_dir, args.scheduler, logger, start_ts, estimator, updater)
                )
            else:
                all_args_list.append((trace, CLUSTER.vc_list[i], args.placer, log_dir, args.scheduler, logger, start_ts))

    with multiprocessing.Pool(processes=process_num) as p:
        results = [p.apply_async(utils.simulate_vc, args_list) for args_list in all_args_list]
        results = [result.get() for result in results]

    if args.sweep:
        for policy in utils.get_sweep_schedulers():
            utils.cluster_concatenate(policy, args.placer, log_dir, args.trace_dir)
    else:
        utils.cluster_concatenate(args.scheduler, args.placer, log_dir, args.trace_dir)
        utils.cluster_analysis(args.placer, log_dir, args.trace_dir)

        """Fast query result"""
        sched_label = args.scheduler + "_consolidate"
        jct_df = pd.read_csv(f"{log_dir}/jct_avg_consolidate.csv", index_col=0)
        jct = jct_df.at["all", sched_label]
        que_df = pd.read_csv(f"{log_dir}/que_avg_consolidate.csv", index_col=0)
        que = que_df.at["all", sched_label]
        logger.info(f"Summary of {args.scheduler}: Avg. JCT {jct}s, Avg. Queue {que}s")

    logger.info(f"Execution Time: {round(time.perf_counter() - code_start, 2)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulator")
    # 'Saturn_Sept', 'Venus_Sept', 'Philly'
    parser.add_argument("-e", "--experiment-name", default="Venus_Sept", type=str, help="Experiment Name")
    parser.add_argument("-t", "--trace-dir", default="./data/Venus", type=str, help="Trace File Directory")
    parser.add_argument("-l", "--log-dir", default="./log", type=str, help="Log Directory")

    parser.add_argument(
        "-s", "--scheduler", default="lucid", choices=utils.get_available_schedulers(), type=str, help="Scheduler Algorithm"
    )
    parser.add_argument(
        "-p", "--placer", default="consolidate", choices=utils.get_available_placers(), type=str, help="Placer Algorithm"
    )

    parser.add_argument(
        "--profiler-auto", default=1, type=int, help="Use default profiling setting, disable below (time, scale, factor)."
    )
    # For ablation study
    parser.add_argument("--profiler-time", default=500, type=int, help="Time limit in profiler, unit: second")
    parser.add_argument("--profiler-scale", default=6, type=int, help="Number of nodes applied in profiler")
    parser.add_argument("--profiler-factor", default=6, type=int, help="Maximum GPU number to be profiled = factor x scale")

    parser.add_argument("--colocate", default=0, type=int, help="Whether to enable GPU sharing")
    parser.add_argument("--pollux-idx", default=None, type=int, help="Index of Pollux Trace")
    parser.add_argument("--sweep", action="store_true", default=False, help="Run All Scheduler Policies in One Time")
    parser.add_argument(
        "-j",
        "--processes",
        type=int,
        default=None,
        help=("Number of processes to use in multiprocessing.Pool" "(use as many as available if not specified)"),
    )
    parser.add_argument("--timeout", default=1209600, type=int, help="Timeout (in seconds), default 14 days")
    parser.add_argument("--num_gpus_per_node", type=int, default=8, help=("Number of GPUs per node"))
    parser.add_argument("--num_cpus_per_node", type=int, default=96, help=("Number of CPU cores per node"))

    args = parser.parse_args()

    main(args)
