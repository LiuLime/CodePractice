import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig, AutoModelForSequenceClassification
from transformers import AlbertTokenizer, AlbertModel
from transformers import TrainingArguments, Trainer, pipeline, AutoTokenizer
import evaluate
import json

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

path = os.getcwd()
data_dict = f'{path}/rotate_tomato'

trainset = pd.read_csv(f'{data_dict}/train.tsv', sep='\t', keep_default_na=False)
trainset = Dataset.from_pandas(trainset)
testset = pd.read_csv(f'{data_dict}/test.tsv', sep='\t', keep_default_na=False)
# keep_default_na=False很重要，否则‘None’会被转换为Nonetype导致分词出错
testset = Dataset.from_pandas(testset)
# %%

model = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokz(x):
    return tokenizer(x['Phrase'], padding=True, truncation=True, return_tensors="pt")


testset_tokz = testset.map(tokz, batched=True)

trainset_tokz = trainset.map(tokz, batched=True)
trainset_tokz = trainset_tokz.rename_columns({'Sentiment': 'labels'})

trainset_tokz_dict = trainset_tokz.train_test_split(0.25, seed=42)

# %%
batch_size = 128
epochs = 10
lr = 8e-5
# 设置度量指标
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


args = TrainingArguments(f'{data_dict}/outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine',  # 余弦退火学习率调度器
                         evaluation_strategy="epoch", per_device_train_batch_size=batch_size,  # 默认的貌似是AdamW优化器
                         per_device_eval_batch_size=batch_size * 2,
                         num_train_epochs=epochs, weight_decay=0.01, report_to='none')

model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=5).to(device)
trainer = Trainer(model, args, train_dataset=trainset_tokz_dict['train'], eval_dataset=trainset_tokz_dict['test'],
                  tokenizer=tokenizer, compute_metrics=compute_metrics)

trainer.train()
trainer.save_model(f'{data_dict}/saved_model')

# %%
model= AutoModelForSequenceClassification.from_pretrained(f'{data_dict}/saved_model', local_files_only=True).to(device)
preds = trainer.predict(testset_tokz).predictions.astype(float)
print(preds)
with open ('preds.json', 'w') as json_file:
    json.dump(preds, json_file)

