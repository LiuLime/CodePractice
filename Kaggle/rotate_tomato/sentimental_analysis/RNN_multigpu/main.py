"""
The entry of the model

copyright@ Liu
"""


import os, sys

path = os.getcwd()
sys.path.append(path)

from model import BiRNN
from parser import get_args
import train
import dataloader
import prediction

import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp


def load_train_objs(args):
    with open(os.path.join(args.data_path, args.train_file), "r") as t:
        trainset = pd.read_csv(t, sep="\t", keep_default_na=False)

    token = dataloader.token(args.tokenizer)
    train_iter, eval_iter = token.dataset_iter(
        trainset, args.train_column, train=True, target_column=args.label_column
    )
    model = BiRNN(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    return train_iter, eval_iter, model, optimizer, criterion


def load_test_objs(args):
    with open(os.path.join(args.data_path, args.test_file), "r") as t:
        testset = pd.read_csv(t, sep="\t", keep_default_na=False)

    token = dataloader.token(args.tokenizer)
    test_iter = token.dataset_iter(testset, args.train_column, train=False)

    # load from checkpoint
    checkpoint = torch.load(args.model_path)
    model = BiRNN(args)
    model.load_state_dict(checkpoint["model_state_dict"])

    return test_iter, model, testset[args.identifier_column]


def main(
    rank: int,
    world_size: int,
    args,
):
    if args.test == False:
        train.ddp_setup(rank, world_size)
        train_iter, eval_iter, model, optimizer, criterion = load_train_objs(args)
        trainer = train.Train(
            args,
            train_iter,
            eval_iter,
            rank,
            model,
            optimizer,
            criterion,
        )
        trainer.train()
        destroy_process_group()

    if args.test == True:
        prediction.ddp_setup(rank, world_size)
        test_iter, model, identifier = load_test_objs(args)
        indices, labels = prediction.preds(
            rank,
            world_size,
            test_iter,
            model,
        )
        destroy_process_group()

    if rank == 0 and labels is not None and indices is not None:
        print(f"{len(labels)} prediction finished")
        pred_reports = {
            args.identifier_column: identifier,
            args.label_column: labels,
            "origin_index": indices,
        }
        pd.DataFrame(pred_reports).to_csv(
            f"{args.output_path}/preds_reports.csv", index=False
        )
        print(f"prediction report saved at {args.output_path}")


if __name__ == "__main__":
    args = get_args()
    # args.world_size = torch.cuda.device_count()
    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
