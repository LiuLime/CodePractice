import json
import argparse
from pathlib import Path
import torch

from dpm import config_

configDL = config_.DLParamConfig()
config_dl = configDL.cnn_params
#
# def load_config(path):
#     with open(path, "r") as j:
#         config = json.load(j)
#     return config


def get_args():
    parent_path = Path(__name__).parent
    config_path = config_.PathConfig(mode='prepr').base_path

    parser = argparse.ArgumentParser(description="distributed training job for deep learning")

    parser.add_argument(
        "--approach",
        default='nn',
        help="The model/algorithm approaches, ['nn'].",
        type=str,
    )
    parser.add_argument(
        "--architecture",
        default='fiveCNN',
        help="The model/algorithm approaches, ['fiveCNN', 'fiveCNN_with_dropout',].",
        type=str,
    )
    parser.add_argument(
        "--mode",
        default='train',
        help="The running mode 'train', 'test', 'preds', 'prepr','lrfinder', default is 'train'",
        type=str,
    )

    # Training configs
    parser.add_argument(
        "--batch_size",
        default=config_dl["batch_size"],
        help="batch size, default=128",
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        default=config_dl["num_epochs"],
        help="training epochs number, default=10.",
        type=int,
    )
    parser.add_argument(
        "--save_every_n_epoch",
        default=config_dl["save_every_n_epoch"],
        help="checkpoint saved every n epoches, default=5",
        type=int,
    )
    parser.add_argument(
        "--patience",
        default=config_dl["patience"],
        help="check evaluate loss decreasement inside patience times, default=10",
        type=int,
    )
    parser.add_argument(
        "--world_size",
        default=config_dl["world_size"],
        help="number of processes started, default=all gpu",
        type=int,
    )

    # Mode config
    # parser.add_argument(
    #     "--test",
    #     default=config_dl["test"],
    #     help="Test on the testset.",
    #     action="store_true",
    # )

    parser.add_argument(
        "--run_from_ckp",
        default=config_dl["run_from_ckp"],
        help="rerun from checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--run_small_set",
        default=config_dl["run_small_set"],
        help="run small dataset for training and evaluating",
        action="store_true",
    )
    parser.add_argument(
        "--early_stopping",
        default=config_dl["early_stopping"],
        help="if evaluate loss did not decrease inside patience times, early stop training",
        action="store_true",
    )
    parser.add_argument(
        "--scaling",
        default=config_dl["scaling"],
        help="conduct feature scaling for train, evaluate, and test set",
        action="store_true",
    )
    parser.add_argument(
        "--save_dataset_idx",
        default=config_dl["save_dataset_idx"],
        help="save random sampled dataset index",
        action="store_true",
    )
    args = parser.parse_args()

    args.model_path = Path(config_path[f"{args.approach}_model_fpath"])
    args.output_path = Path(config_path[f"{args.approach}_output_path"])
    args.metric_path = Path(config_path[f"{args.approach}_metric_fpath"])
    return args
