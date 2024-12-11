import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


import datasets.busi
import datasets.lungcolon
import datasets.chmnist
import datasets.covid
import datasets.btmri
import datasets.ctkidney
import datasets.kvasir
import datasets.retina
import datasets.kneexray
import datasets.dermamnist 
import datasets.octmnist

import trainers.zeroshot
import trainers.coop_clip
import trainers.coop_biomedclip
import trainers.coop_pubmedclip
import trainers.coop_pmcclip
import trainers.cocoop_clip
import trainers.cocoop_biomedclip
import trainers.cocoop_pubmedclip
import trainers.cocoop_pmcclip
import trainers.kgcoop_clip
import trainers.kgcoop_biomedclip
import trainers.kgcoop_pubmedclip
import trainers.kgcoop_pmcclip
import trainers.prograd_clip
import trainers.prograd_biomedclip
import trainers.prograd_pubmedclip
import trainers.prograd_pmcclip
import trainers.biomedcoop_clip
import trainers.biomedcoop_biomedclip
import trainers.biomedcoop_pubmedclip
import trainers.biomedcoop_pmcclip



def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COCOOP.CSC = False  # class-specific context
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.BIOMEDCOOP.CSC = False  # class-specific context
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.5
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50

    cfg.TRAINER.KGCOOP = CN()
    cfg.TRAINER.KGCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KGCOOP.CSC = False  # class-specific context
    cfg.TRAINER.KGCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.KGCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.KGCOOP.W = 1.0

    cfg.TRAINER.PROGRAD = CN()
    cfg.TRAINER.PROGRAD.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROGRAD.CSC = False  # class-specific context
    cfg.TRAINER.PROGRAD.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROGRAD.N_CTX = 4  # number of context vectors
    cfg.TRAINER.PROGRAD.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.PROGRAD.GM = False
    cfg.TRAINER.PROGRAD.NAME = ""
    cfg.TRAINER.PROGRAD.ALPHA = 0.
    cfg.TRAINER.PROGRAD.T = 1.
    cfg.TRAINER.PROGRAD.LAMBDA = 1.

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)


    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
