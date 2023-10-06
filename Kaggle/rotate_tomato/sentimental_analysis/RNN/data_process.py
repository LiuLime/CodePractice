import torch
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader

class tokenization():

    def __init__(self, tokenizer, batch_size=24):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def small_set(self, set, seed, range_num):
        return set.shuffle(seed=seed).select(range(range_num))

    def dataset_iter(self, dataset, content_column, train=True, target_column=None, batch_size = 24):
        tokz = lambda x: self.tokenizer(x[content_column], padding='max_length', truncation=True, return_tensors="pt")

        if train is True:
            if target_column is None:
                raise TypeError('target_column args is missing')
            else:
                trainset = Dataset.from_pandas(dataset)
                trainset_tokz = trainset.map(tokz, batched=True).rename_columns({target_column: 'labels'})
                trainset_tokz.set_format('torch')
                # print('train_tokz_ids',trainset_tokz['input_ids'].shape)

                trainset_tokz_dict = trainset_tokz.train_test_split(0.2, seed=42)
                train_tokz_iter = TensorDataset(trainset_tokz_dict['train']['input_ids'],
                                                trainset_tokz_dict['train']['labels'])  # 返回tuple格式的dataset
                train_tokz_iter = DataLoader(train_tokz_iter, shuffle=True, batch_size=batch_size, )
                evl_tokz_iter = TensorDataset(trainset_tokz_dict['test']['input_ids'],
                                              trainset_tokz_dict['test']['labels'])
                evl_tokz_iter = DataLoader(evl_tokz_iter, shuffle=True, batch_size=batch_size)
                return train_tokz_iter, evl_tokz_iter
        else:
            testset = Dataset.from_pandas(dataset)
            testset_tokz = testset.map(tokz)
            testset_tokz.set_format('torch')
            i = torch.tensor(range(len(testset_tokz['input_ids'])))
            test_tokz_iter = TensorDataset(i, testset_tokz['input_ids'])
            test_tokz_iter = DataLoader(test_tokz_iter, batch_size=batch_size)
            return test_tokz_iter
