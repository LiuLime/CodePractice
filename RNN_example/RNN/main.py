'''
The entry of the model

copyright@ Liu
'''

import argparse
import os, sys

path = os.getcwd()
sys.path.append(path)
from model import BiRNN
import train
import data_process
import prediction

import torch
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path



def main():
    parent_path = Path(__name__).parent  # main.py所在文件夹
    relative_data_path = Path("../data/rotate_tomato/")
    relative_output_path = Path("../output/rotate_tomato/")


    # Parse the cmd arguments
    parser = argparse.ArgumentParser()

    # Path and file configs
    parser.add_argument('--data_path', default=relative_data_path, help='The dataset path.', type=str)
    parser.add_argument('--model_path', default=relative_output_path/'BiRNN_params.pt', help='The model will be saved to this path.',
                        type=str)
    parser.add_argument('--output_path', default=relative_output_path,
                        help='The predictions will be saved to this path.',
                        type=str)
    parser.add_argument('--train_file', default='train.tsv', type=str)
    parser.add_argument('--val_file', default=None, type=str)
    parser.add_argument('--test_file', default='test.tsv', type=str)
    parser.add_argument('--content', default='Phrase', help='The content column for training', type=str)
    parser.add_argument('--label_column', default='labels', help='The label column for training', type=str)

    # Model configs
    parser.add_argument('--tokenizer_name', default='distilbert-base-uncased',
                        help='pre-trained tokenizer, default=distilbert-base-uncased', type=str)

    parser.add_argument('--embedding_dim', default=128, help='Tokens will be embedded to a vector.', type=int)
    parser.add_argument('--hidden_dim', default=256, help='The hidden state dim of BiLSTM.', type=int)
    parser.add_argument('--output_dim', default=5, help='The output dim of BiLSTM-> num of label class', type=int)
    parser.add_argument('--num_layers', default=2, help='The number of LSTM layers.', type=int)

    # Optimizer config
    parser.add_argument('--lr', default=1e-4, help='Learning rate of the optimizer.', type=float)

    # Training configs
    parser.add_argument('--train_batch_size', default=16, help='Batch size for training.', type=int)
    parser.add_argument('--test_batch_size', default=16, help='Batch size for testing.', type=int)
    parser.add_argument('--num_epochs', default=10, help='epochs number for training.', type=int)
    # parser.add_argument('--eval_steps', default=200, help='Total number of training epochs to perform.', type=int)

    # Device config
    parser.add_argument('--gpu', default=0, type=int)

    # Mode config
    parser.add_argument('--test', help='Test on the testset.', action='store_true')

    args = parser.parse_args()

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
