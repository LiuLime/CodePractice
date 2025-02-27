import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertModel, AutoModelForSequenceClassification
import evaluate
import json

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")
device = 'cpu'
path = os.getcwd()

trainset = pd.read_csv(f'/Users/liuyuting/Git_Liu/CodePractice/Kaggle/sentimental_analysis/data/train.tsv', sep='\t', keep_default_na=False)
testset = pd.read_csv(f'/Users/liuyuting/Git_Liu/CodePractice/Kaggle/sentimental_analysis/data/test.tsv', sep='\t', keep_default_na=False)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokz(x):
    # print(type(x['Phrase']))  # x['Phrase']是list
    return tokenizer(x['Phrase'], padding='max_length', truncation=True, return_tensors="pt")


trainset_tokz = Dataset.from_pandas(trainset).map(tokz, batched=True).rename_columns({'Sentiment': 'labels'})
testset_tokz = Dataset.from_pandas(testset).map(tokz, batched=True)

small_train_tokz = trainset_tokz.shuffle(seed=42).select(range(1000))
small_test_tokz = testset_tokz.shuffle(seed=42).select(range(1000))
small_train_tokz.set_format('torch')
small_test_tokz.set_format('torch')

small_train_tokz = small_train_tokz.remove_columns(['PhraseId', 'SentenceId', 'Phrase'])
small_train_tokz_dict = small_train_tokz.train_test_split(0.2, seed=42)

# %%
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

train_dataloader = DataLoader(small_train_tokz_dict['train'], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_train_tokz_dict['test'], batch_size=8)
#%%
model = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=5).to(device)

# optimizer and lr
optimizer = AdamW(model.parameters(), lr=5e-5)
# Create the default learning rate scheduler from Trainer
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# %% Training loop
progress_bar = tqdm(range(num_training_steps))

model.train()  # 把model设置为training模式，开启dropout或batch normalization，只需在training之前设置一次

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # 这步操作将batch中所有的键值对都转移到device上
        outputs = model(**batch)  # 前向传播
        loss = outputs.loss  # 损失值被从outputs提取出来。Note：所有transformers model都有默认任务相关的损失函数，无需自己指定但也可以覆盖
        loss.backward()  # 反向传播

        optimizer.step()  # 更新权重
        lr_scheduler.step()  # 调整学习率
        optimizer.zero_grad()  # 清零梯度，为下一次训练做准备
        progress_bar.update(1)
# 用cpu可以跑完，但是用mps跑就会出现泄漏.
PATH = f'{path}/torchmodel.pt'
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, PATH)
# %%
import evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():  # evaluation模式关闭梯度下降
        outputs = model(**batch)

    logits = outputs.logits  # logits值non-normalized raw data generated by model
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])  # add_batch累积计算batch性能指标，以用于最后compute最终整体的性能

metric.compute()  # compute 最终整体性能
