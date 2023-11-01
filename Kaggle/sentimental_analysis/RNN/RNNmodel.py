import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset
import pandas as pd
from transformers import DistilBertTokenizer, get_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from pathlib import Path
import evaluate
import numpy as np
from d2l import torch as d2l

device = 'cpu'
path = os.getcwd()
parent_path = Path(__name__).parent
relative_data_path = Path("../../sentimental_analysis/data/rotate_tomato")
full_data_path = parent_path / relative_data_path
full_data_path = full_data_path.resolve()
# %% Read

trainset = pd.read_csv(os.path.join(full_data_path, 'train.tsv'), sep='\t', keep_default_na=False)
trainset = Dataset.from_pandas(trainset)
# %% Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokz(x):
    return tokenizer(x['Phrase'], padding='max_length', truncation=True, return_tensors="pt")


small_train = trainset.shuffle(seed=42).select(range(1000))
small_train_tokz = small_train.map(tokz, batched=True).rename_columns({'Sentiment': 'labels'})
small_train_tokz.set_format('torch')

# %% Define train_iter
small_train_tokz_dict = small_train_tokz.train_test_split(0.2, seed=42)

small_train_tokz_iter = TensorDataset(small_train_tokz_dict['train']['input_ids'],
                                      small_train_tokz_dict['train']['labels'])  # 返回tuple格式的dataset
small_train_tokz_iter = DataLoader(small_train_tokz_iter, shuffle=True, batch_size=8)
small_evl_tokz_iter = TensorDataset(small_train_tokz_dict['test']['input_ids'], small_train_tokz_dict['test']['labels'])
small_evl_tokz_iter = DataLoader(small_evl_tokz_iter, shuffle=True, batch_size=8)


# %% 定义LSTM模型

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, output_dim, ):
        super(BiRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(4 * hidden_dim, output_dim)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):  # input x (batch_size, seq_length)

        embeds = self.embedding(x)  # in(batch_size, seq_length)->out(batch_size, seq_length, embedding_dim)
        print(x)
        print(x.shape)
        self.encoder.flatten_parameters()  # 优化参数存储

        out, (h, c) = self.encoder(embeds)  # in(batch_size, seq_length, embedding_dim)->out=(batch_size, sequence_length,
        # 2 * hidden_dim),h=(hidden_state, cell_state)&state=(2 * n_layers, batch_size, hidden_dim)

        first_timestep = out[:, 0, :]  # 第一个时间步的[batch_size, 2*hidden_dim]
        last_timestep = out[:, -1, :]  # 最后时间步的[batch_size, 2*hidden_dim]
        out2 = torch.cat((first_timestep, last_timestep), dim=1)  # 拼接第一和最后一个时间步的2*hidden_dim，seq_length维度消失，out(
        # batch_size, 4 * hidden_dim)

        out2 = self.decoder(out2)  # in(batch_size, 4 * hidden_dim) -> out(batch_size, output_dim)
        soft_out = self.soft(out2)

        return soft_out


# %% 实例化模型、定义损失函数和优化器
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
vocab_size = tokenizer.vocab_size
embedding_dim = 100
hidden_dim = 256
n_layers = 2
output_dim = 5

model = BiRNN(vocab_size, embedding_dim, hidden_dim, n_layers, output_dim).to(device)


# %%
# 网络权重初始化
def init_weights(m):
    if type(m) == nn.Linear:  # linear层的权重存储在.weight里
        nn.init.xavier_uniform_(m.weight)  # Xavier初始化是一种常用的权重初始化方法，旨在使神经网络在训练时更稳定和更快速地收敛
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:  # _flat_weights_names是一个私有属性，用于存储模型中所有权重参数的名称列表
            # print(param)
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


model.apply(init_weights)

# "mean"（默认值）：损失函数计算所有样本的损失值并返回平均值。用于批量训练。"sum"：计算所有样本的损失值并返回总和。"none"：损失函数计算每个样本的损失值，用于自定义损失函数
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=5e-5)

print('Model:', model)
# %% 训练模型

epochs = 10
num_training_steps = epochs * len(small_train_tokz_iter)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for i, (input_id, label) in enumerate(small_train_tokz_iter):
        input_id = input_id.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        out = model(input_id)  # 前向传播
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        lr_scheduler.step() # 调整学习率
        total_train_loss += loss.item()
        progress_bar.update(1)

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs,
                                                                     epoch * len(small_train_tokz_iter) + i + 1,
                                                                     num_training_steps, loss.item()))
    avg_train_loss = total_train_loss / len(small_train_tokz_iter)

    # validation
    model.eval()  # 设置模型为评估模式
    eval_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in small_evl_tokz_iter:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()

            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()

    avg_val_loss = eval_loss / len(small_evl_tokz_iter)
    val_accuracy = 100. * correct / len(small_evl_tokz_iter.dataset)

    print(
        f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

torch.save(model.state_dict(), f'{path}/RNNmodel_params.ckpt')
# %% prediction

batch_size = 20
device = 'cpu'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

iterm='Phrase'
def tokz(x):

    return tokenizer(x['Phrase'], padding='max_length', truncation=True, return_tensors="pt")


testset = pd.read_csv(f'{path}/test.tsv', sep='\t', keep_default_na=False)
testset = Dataset.from_pandas(testset)

test_tokz=testset.map(tokz, batched=True)
test_tokz.set_format('torch', device=device)
i = torch.tensor(range(len(test_tokz['input_ids'])))
test_tokz_iter = TensorDataset(i, test_tokz['input_ids'])
test_input = DataLoader(test_tokz_iter, batch_size=batch_size)

# %%
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
vocab_size = tokenizer.vocab_size
embedding_dim = 100
hidden_dim = 256
n_layers = 2
output_dim = 5

model_load = BiRNN(vocab_size, embedding_dim, hidden_dim, n_layers, output_dim).to(device)
model_load.load_state_dict(torch.load(f'{path}/RNNmodel_params.ckpt'))
model_load.eval()
labels=[]
with torch.no_grad():
    for i, in_id in test_input:
        in_id=in_id.to(device)
        preds_logit = model_load(in_id)

        label = torch.argmax(preds_logit, dim=-1).to('cpu').detach().numpy()  # .detach().numpy()的组合用于输出从PyTorch张量转换为NumPy数组
        labels.extend(label)
pred_reports={'PhraseId': testset['PhraseId'],
              'Sentiment': labels}
pd.DataFrame(pred_reports).to_csv(f'{path}/Sentiment_analysis_RNN.csv', index=False)
