import os

import logging
import random

import numpy as np
import torch
from transformers import BertTokenizer

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def official_f1(args):
    # Run the perl script
    try:
        cmd = "perl {0}/semeval2010_task8_scorer-v1.2.pl {1}/proposed_answers.txt {0}/answer_keys.txt > {1}/result.txt".format(
            args.data_dir, args.output_dir
        )
        os.system(cmd)
    except:
        raise Exception("perl is not installed or proposed_answers.txt is missing")

    with open(os.path.join(args.output_dir, "result.txt"), "r", encoding="utf-8") as f:
        macro_result = list(f)[-1]
        macro_result = macro_result.split(":")[1].replace(">>>", "").strip()
        macro_result = macro_result.split("=")[1].strip().replace("%", "")
        macro_result = float(macro_result) / 100

    return macro_result

def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(args, preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(args, preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(args, preds, labels, average="macro", ):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
        "f1": official_f1(args),
    }