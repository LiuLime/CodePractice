
import argparse
from pathlib import Path

def load_config(path):
    with open (path, "r") as j:
        config = json.load(j)
    return config


def get_args():

    parent_path = Path(__name__).parent 
    config = load_config("config.json")

    relative_data_path = Path(config["data_path"])
    relative_output_path = Path(config["output_path"])
    relative_model_path =Path(config["model_path"])

    parser = argparse.ArgumentParser(description="distributed training job for BiRNN")

    # Path and file configs
    parser.add_argument('--data_path', default=relative_data_path, help='The dataset path.', type=str)
    parser.add_argument('--model_path', default=relative_model_path, help='The model will be saved to this path.',
                        type=str)
    parser.add_argument('--output_path', default=relative_output_path,
                        help='The predictions will be saved to this path.',
                        type=str)
    parser.add_argument('--train_file', default=config["train_file"], help='default=train.tsv',type=str)
    parser.add_argument('--val_file', default=None, help='validation file, default=None',type=str)
    parser.add_argument('--test_file', default=config["test_file"], help='default=test.tsv', type=str)
    parser.add_argument('--content', default=config["content"], help='The content column, default=Phrase', type=str)
    parser.add_argument('--label_column', default=config["label_column"], help='The label column, default=labels', type=str)

    # Model configs
    parser.add_argument('--tokenizer_name', default=config["tokenizer_name"],
                        help='pre-trained tokenizer, default=distilbert-base-uncased', type=str)

    parser.add_argument('--embedding_dim', default=config["embedding_dim"], help='Tokens will be embedded to a vector, default=128', type=int)
    parser.add_argument('--hidden_dim', default=config["hidden_dim"], help='The hidden state dim of BiLSTM. default=256', type=int)
    parser.add_argument('--output_dim', default=config["output_dim"], help='output dim of BiLSTM-> num of label class, default=5', type=int)
    parser.add_argument('--num_layers', default=config["num_layers"], help='number of LSTM layers, default=2', type=int)

    # Optimizer config
    parser.add_argument('--lr', default=config["lr"], help='Learning rate of the optimizer,default=1e-4', type=float)

    # Training configs
    parser.add_argument('--train_batch_size', default=config["train_batch_size"], help='training batch size, default=64', type=int)
    parser.add_argument('--test_batch_size', default=config["test_batch_size"], help='test batch size, default=64', type=int)
    parser.add_argument('--num_epochs', default=config["num_epochs"], help='training epochs number, default=10.', type=int)

    # Device config
    parser.add_argument('--gpu', default=config["gpu"], type=int)

    # Mode config
    parser.add_argument('--test', default=config["test"],help='Test on the testset.', action='store_true')

    args = parser.parse_args()

    return args