'''
The entry of the model

copyright@ Liu
'''

import argparse
import os, sys

path = os.getcwd()
sys.path.append(path)
from model import BiRNN
from parser import get_args
import train
import data_process
import prediction

import torch
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path



def main():

    args = get_args()
    # Specify the device. If you have a GPU, the training process will be accelerated.
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    args.vocab_size = args.tokenizer.vocab_size

    # Load the datasets
    token = data_process.tokenization(args.tokenizer)
    model = BiRNN(args).to(args.device)

    if not args.test:
        with open(os.path.join(args.data_path, args.train_file), 'r') as t:
            trainset = pd.read_csv(t, sep='\t', keep_default_na=False)

        train_iter, evaluate_iter = token.dataset_iter(trainset, args.content, train=True,
                                                       target_column=args.label_column)
        t = train.Train(args, train_iter, evaluate_iter)
        t.BiRNN_train(model)
    else:
        with open(os.path.join(args.data_path, args.test_file), 'r') as t:
            testset = pd.read_csv(t, sep='\t', keep_default_na=False)

        model.load_state_dict(torch.load(args.model_path))
        test_iter = token.dataset_iter(testset, args.content, train=False, target_column=args.label_column)
        labels = prediction.Prediction.BiRNN_preds(model, test_iter, args.device)
        pred_reports = {args.content: testset[args.content],
                        args.label_column: labels}
        pd.DataFrame(pred_reports).to_csv(f'{args.output_path}/preds_reports.csv', index=False)


if __name__ == '__main__':
    main()
