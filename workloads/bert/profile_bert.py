from __future__ import print_function
import argparse
import timeit
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import sys
import numpy as np
import os
import pandas as pd
import torchvision
import time
from torch.nn import DataParallel
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import workloads.settings as settings

# Benchmark settings
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# print("Model type selected in the list: " + ", ".join(MODEL_TYPES))

parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
parser.add_argument("--gpu", default=1, type=int, help="GPU id to use. Only work when use single gpu.")
parser.add_argument(
    "--num-warmup-batches", type=int, default=1, help='number of warm-up batches that don"t count towards benchmark'
)
parser.add_argument("--num-batches-per-iter", type=int, default=1, help="number of batches per benchmark iteration")
parser.add_argument("--num-iters", type=int, default=1, help="number of benchmark iterations")
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--warmup_epoch', type=int, default=20, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=160, help='number of training benchmark epochs')
#####################################
parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
parser.add_argument(
    "--model_name_or_path",
    default='bert-base-uncased',
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
# Other parameters
parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    help="The input data dir. Should contain the .json files for the task."
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--train_file",
    default='/home/mzhang/data/SQUAD_DIR/train-v1.1.json',
    type=str,
    help="The input training file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--predict_file",
    default=None,
    type=str,
    help="The input evaluation file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
)
parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
)
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)

parser.add_argument(
    "--version_2_with_negative",
    action="store_true",
    help="If true, the SQuAD examples contain some that do not have an answer.",
)
parser.add_argument(
    "--null_score_diff_threshold",
    type=float,
    default=0.0,
    help="If null_score - best_non_null is greater than the threshold predict null.",
)

parser.add_argument(
    "--max_seq_length",
    default=384,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
    "longer than this will be truncated, and sequences shorter than this will be padded.",
)
parser.add_argument(
    "--doc_stride",
    default=128,
    type=int,
    help="When splitting up a long document into chunks, how much stride to take between chunks.",
)
parser.add_argument(
    "--max_query_length",
    default=64,
    type=int,
    help="The maximum number of tokens for the question. Questions longer than this will "
    "be truncated to this length.",
)
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
)
parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
)

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--n_best_size",
    default=20,
    type=int,
    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
)
parser.add_argument(
    "--max_answer_length",
    default=30,
    type=int,
    help="The maximum length of an answer that can be generated. This is needed because the start "
    "and end predictions are not conditioned on one another.",
)
parser.add_argument(
    "--verbose_logging",
    action="store_true",
    help="If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.",
)
parser.add_argument(
    "--lang_id",
    default=0,
    type=int,
    help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
)

parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
)
parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

args = parser.parse_args()

args.train_file = settings.data_dir + 'SQUAD_DIR/train-v1.1.json'


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
    examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
    )

    if output_examples:
        return dataset, examples, features
    return dataset


def benchmark_bert(model_name, batch_size, mixed_precision, gpu_id, bench_list, warm_signal):
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model.to(device)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Train
    def benchmark_step():
        iter_num = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Warm-up: previous 10 iters
            if iter_num == args.warmup_epoch-1:
                warm_signal.value = 1
                t_start = time.time()
            # Benchmark: 50 iters
            if iter_num == args.warmup_epoch+args.benchmark_epoch-1:
                t_end = time.time()
                t_pass = t_end - t_start
                break
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(device)}
                    )
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    loss = outputs[0]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**inputs)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
            iter_num += 1
        return t_pass

    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass = benchmark_step()
    img_sec = args.benchmark_epoch * batch_size / t_pass
  
    # Results
    bench_list.append(img_sec)
