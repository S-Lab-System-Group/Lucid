# Artifact for Lucid
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7275326.svg)](https://doi.org/10.5281/zenodo.7275326)

This repository contains the artifact for our ASPLOS '23 paper "*Lucid: A Non-Intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs*". It includes following parts:

+ `simulation`: It contains code and data for reproducing key results in our paper.

+ `workloads`: The Pytorch implementation of 14 different workloads used in experiments. 

+ `profile`: It contains the code to collect traces of each training job type.

# Getting Started

### Results Reproduction (for ASPLOS '23 Artifact Evaluation)
`simulation` (adopted from [Helios](https://github.com/S-Lab-System-Group/HeliosArtifact)) contains instructions for reproducing the `Venus` cluster experiments shown in Section 4. These scripts have been tested on Ubuntu 20.04 with Python 3.9.

#### 0. Structure

The contents inside `simulation` folder are summarized as follows:

- **data/** contains `Venus` cluster job trace and cluster configuration used for evaluation.
- **analyzer/** contains the *Packing Analyze Model* and profiled workloads information used in our experiment.
- **estimator/** contains the *Workload Estimate Model* and job duration estimation for both Lucid and QSSF.
- **plot/** contains notebook for visualizing experiment results.
- **policy/** contains implementations of the Lucid scheduling policy, and baseline policies including FIFO, SJF, QSSF, Tiresias. 
- **predictor/** contains the *Throughput Predict Model* and cluster throughput estimation in Venus September.
- **profiler/** contains the Least-GPU-First and Auto-Scaling Profiler implementation for Lucid.
- **cluster.py**, **job.py** and **updater.py** contain implementations of the GPU cluster and workload logic.
- **simulator.py** is the main entry of the simulator.


#### 1. Environment Preparation

We suggest using a conda environment to install the dependencies:

```bash
conda create -n lucid python=3.9
conda activate lucid
cd simulation
pip install -r requirements.txt
```

Besides, we recommend execute Jupyter notebook (`.ipynb`) files with **VSCode** or **JupyterLab** (`conda install jupyterlab`).

#### 2. Lucid Model Training and Interpretation

We train *Throughput Predict Model* as a reproduction example. Please follow below steps: 

+ Enter `predictor` folder and open `predictor.ipynb` file

+ Run all cells inside the notebook. It contains the interpretable model (Primo EBM) used in Lucid and other ML baselines (LightGBM, XGBoost, Random Forest, DNN).

+ **Table 7: Interpretable Model Performance**: Check `Result Comparison` cell, the MAE scores of all baselines are listed.

+ **Figure 13 (a): Throughput Predict Performance**: Check `Prediction Visualization` cell (or `Venus_throughput.pdf` output file), both the real and predicted throughput are plotted. Generated figures should have similar patterns as the paper. The difference is because we release the *Venus Job* throughput prediction code but we plot *Saturn Job* throughput prediction in our paper.

+ **Figure 7 (a)(b): Global Model Interpretation and Learned Shape Function**: Check `Model Interpretation` cell (or `interpret_Venus_throughput.pdf` & `interpret_Venus_shapefunc.pdf` output files). Generated figures should have similar patterns as the paper. The difference is because we release the *Venus Job* throughput prediction code but we plot *Saturn GPU* throughput prediction in our paper.


More model training codes are also provided (`estimator/estimator_lucid.ipynb` and `analyzer/analyzer.py`).


#### 3. Reproduce Baseline Results

Use the following command to run all baselines simultaneously

```bash
cd simulation
python simulator.py --sweep 
```

The output of this script looks like this:
```
2022 Oct 08 14:32:57 | MainProcess | Total Job Number in Cluster Training: 23859
2022 Oct 08 14:32:59 | ForkPoolWorker-1 | vcEwI | Time: 13220000 | Total Job: 7603 | End job: 13 | Running job: 2 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-2 | vcWoR | Time: 13220000 | Total Job: 2826 | End job: 0 | Running job: 0 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-1 | vcEwI | Time: 13230000 | Total Job: 7603 | End job: 120 | Running job: 4 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-2 | vcWoR | Time: 13230000 | Total Job: 2826 | End job: 0 | Running job: 1 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-1 | vcEwI | Time: 13240000 | Total Job: 7603 | End job: 120 | Running job: 4 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-3 | vcHvQ | Time: 13220000 | Total Job: 2654 | End job: 1 | Running job: 1 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-2 | vcWoR | Time: 13240000 | Total Job: 2826 | End job: 0 | Running job: 1 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-1 | vcEwI | Time: 13250000 | Total Job: 7603 | End job: 121 | Running job: 4 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-4 | vcvGl | Time: 13220000 | Total Job: 1452 | End job: 0 | Running job: 0 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-2 | vcWoR | Time: 13250000 | Total Job: 2826 | End job: 0 | Running job: 2 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-3 | vcHvQ | Time: 13230000 | Total Job: 2654 | End job: 2 | Running job: 0 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-1 | vcEwI | Time: 13260000 | Total Job: 7603 | End job: 162 | Running job: 9 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-5 | vc8Gr | Time: 13220000 | Total Job: 710 | End job: 0 | Running job: 0 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-4 | vcvGl | Time: 13230000 | Total Job: 1452 | End job: 1 | Running job: 2 | Pending job: 0
2022 Oct 08 14:32:59 | ForkPoolWorker-5 | vc8Gr | Time: 13230000 | Total Job: 710 | End job: 0 | Running job: 1 | Pending job: 0
```

#### 4. Reproduce Lucid Results

Similarly, use the following command to run all baselines simultaneously

```bash
python simulator.py -s lucid
```

The output of this script looks like this:
```
2022 Oct 08 14:45:07 | MainProcess | Total Job Number in Cluster Training: 23859
2022 Oct 08 14:45:08 | MainProcess | profvc | Time: 13220000 | Total Job: 23859 | End job: 17 | Running job: 1 | Pending job: 0 | Avail Nodes: 2
2022 Oct 08 14:45:08 | MainProcess | profvc | Time: 13230000 | Total Job: 23859 | End job: 134 | Running job: 0 | Pending job: 0 | Avail Nodes: 2
2022 Oct 08 14:45:08 | MainProcess | profvc | Time: 13240000 | Total Job: 23859 | End job: 134 | Running job: 0 | Pending job: 0 | Avail Nodes: 2
2022 Oct 08 14:45:08 | MainProcess | profvc | Time: 13250000 | Total Job: 23859 | End job: 136 | Running job: 0 | Pending job: 0 | Avail Nodes: 2
2022 Oct 08 14:45:08 | MainProcess | profvc | Time: 13260000 | Total Job: 23859 | End job: 249 | Running job: 3 | Pending job: 4 | Avail Nodes: 1
2022 Oct 08 14:45:08 | MainProcess | profvc | Time: 13270000 | Total Job: 23859 | End job: 385 | Running job: 3 | Pending job: 2 | Avail Nodes: 1
2022 Oct 08 14:45:08 | MainProcess | profvc | Time: 13280000 | Total Job: 23859 | End job: 589 | Running job: 2 | Pending job: 0 | Avail Nodes: 1
2022 Oct 08 14:45:08 | MainProcess | profvc | Time: 13290000 | Total Job: 23859 | End job: 780 | Running job: 2 | Pending job: 0 | Avail Nodes: 2
```

After the program is executed, you can check the result in the `log` folder. The job log and time sequence of each VC are provided separately.


#### 5. Visualize the Key Results

We provide simulation analysis and plot scripts to generate the figures shown in our paper. Please follow below steps: 

+ Enter `plot` folder and open `result_plot.ipynb` file

+ Run all cells inside the notebook. 

+ **Table 4: Scheduling Performance**: Check `Table 4: Result Summary` cell (or `result_summary.csv` output file), the Average JCT, Average Queuing Delay and Queuing Delay 99.9 Quantile of all policies are listed.

+ **Table 5: Scheduling Performance (workload analysis)**: Check `Table 5: Result Summary of Different Scales of Workloads` cell, the Average JCT, Average Queuing Delay of large and small jobs are listed.


+ **Figure 8: CDF of JCT**: Check `Plot Result 8: JCT` cell (or `result_cdf_jct.pdf` output file), JCT CDF of all policies are plotted.

+ **Figure 9: Queue Time in each VC**: Check `Plot Result 9: Queue Time in each VC` cell (or `result_bar_queue.pdf` output file), queuing delay of all policies are plotted.



# Workloads Profiling

This part `profile` contains code for profiling metrics of multiple workloads.

## Directory
Note that `./result/` will be created when `main_co.py` or `main_single.py` is launched.

## Basic Usage
Run `main_co.py` will generate the colocated jobs' metrics under `./result/colocate`. Run `main_single.py` will generate single jobs' metrics under `./result/`. Some specific settings can be set in each workload's profiling file, e.g.`profile_cifar.py`. The output will be like this:
```
imagenet + imagenet
co-locate:
==> Training mobilenet_v3_small model with 32 batchsize, 0 mp..
==> Training mobilenet_v3_small model with 32 batchsize, 0 mp..
co-locate:
==> Training mobilenet_v3_small model with 32 batchsize, 0 mp..
==> Training mobilenet_v3_small model with 32 batchsize, 1 mp..
co-locate:
==> Training mobilenet_v3_small model with 32 batchsize, 1 mp..
==> Training mobilenet_v3_small model with 32 batchsize, 1 mp..
imagenet + cifar10
co-locate:
Files already downloaded and verified
==> Training ResNet18 model with 32 batchsize, 0 mp..
==> Training mobilenet_v3_small model with 32 batchsize, 0 mp..
...
```

## Datasets
The data path storing all datasets is specified in `./workloads/settings.py` as `data_dir`. You can also specify the total runtime of some workloads by changing `total_runtime`.


- CIFAR-10: The cifar10 dataset will be downloaded automatically(if not exist) when `./workloads/cifar/profile_cifar.py` is run.

- ImageNet: The dataset is generated automatically in `./workloads/imagenet/profile_imagenet.py`.

- LSUN: The dataset is generated automatically in `./workloads/dcgan/profile_dcgan.py`. You can change the custom image size of generated data via `--imageSize`. The default value is 64.

- ShapeNet: Use the following command to download dataset under directory `data_dir/shapenetcore/`:

    ```bash
    wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
    unzip shapenetcore_partanno_segmentation_benchmark_v0.zipcollect_metric/workloadspointnet.pytorch.

- SQuAD: The data can be downloaded with the following link and should be saved under `data_dir/SQUAD_DIR/` directory.

    [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)

- Wikitext2: The dataset can be downloaded from 

    [wikitext-2](https://github.com/pytorch/examples/tree/main/word_language_model/data/wikitext-2)

    File `test.txt`, `train.txt` and `valid.txt` should be saved in `data_dir/wikitext-2/` directory.

- Multi30k: First download the Moses tokenizer(http://www.statmt.org/moses/) for data preparation:
    ```bash
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
    sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
    ```
    These files should be downloaded in `./workloads/translation/`.

    Then download data in `data_dir/multi30k/`:
    ```bash
    mkdir -p data/multi30k
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C data/multi30k && rm mmt16_task1_test.tar.gz
    ```
    Preprocess the data:
    ```bash
    for l in en de; do for f in ~/data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
    for l in en de; do for f in ~/data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
    python preprocess.py -train_src ~/data/multi30k/train.en.atok -train_tgt ~/data/multi30k/train.de.atok -valid_src ~/data/multi30k/val.en.atok -valid_tgt ~/data/multi30k/val.de.atok -save_data ~/data/multi30k.atok.low.pt
    ```
    Referenced from: https://github.com/Eathoublu/attention-is-all-you-need-pytorch.

- MovieLens: Use the following command to download the dataset in `data_dir/ml-1m/`:
    ```bash
    wget https://github.com/hexiangnan/neural_collaborative_filtering/raw/master/Data/ml-1m.test.negative
    wget https://github.com/hexiangnan/neural_collaborative_filtering/raw/master/Data/ml-1m.test.rating
    wget https://github.com/hexiangnan/neural_collaborative_filtering/raw/master/Data/ml-1m.train.rating
    ```