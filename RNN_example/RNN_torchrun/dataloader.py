import torch
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class token:
    def __init__(self, tokenizer, batch_size=24):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def small_set(self, set:Dataset, seed, range_num):
        return  set.shuffle(seed=seed).select(range(range_num))

    def dataset_iter(
        self,
        dataset:Dataset,
        train_column,
        train=True,
        target_column=None,
        batch_size: int = 24,
    ) -> DataLoader:
        tokz = lambda x: self.tokenizer(
            x[train_column],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if train is True:
            if target_column is None:
                raise TypeError("target_column args is missing")

            # trainset = Dataset.from_pandas(dataset)
            trainset_tokz = dataset.map(tokz, batched=True).rename_columns(
                {target_column: "labels"}
            )
            trainset_tokz.set_format("torch")

            trainset_tokz_dict = trainset_tokz.train_test_split(0.2, seed=42)

            train_tokz_iter = TensorDataset(
                trainset_tokz_dict["train"]["input_ids"],
                trainset_tokz_dict["train"]["labels"],
            )
            evl_tokz_iter = TensorDataset(
                trainset_tokz_dict["test"]["input_ids"],
                trainset_tokz_dict["test"]["labels"],
            )
            train_tokz_iter = DataLoader(
                dataset=train_tokz_iter,
                shuffle=False,
                batch_size=batch_size,
                pin_memory=True,
                sampler=DistributedSampler(
                    train_tokz_iter, shuffle=True, drop_last=True
                ),
            )
            evl_tokz_iter = DataLoader(
                evl_tokz_iter,
                shuffle=False,
                batch_size=batch_size,
                pin_memory=True,
                sampler=DistributedSampler(
                    train_tokz_iter, shuffle=True, drop_last=True
                ),
            )

            return train_tokz_iter, evl_tokz_iter

        if train is False:
            # testset = Dataset.from_pandas(dataset)
            testset_tokz = dataset.map(tokz, batched=True)
            testset_tokz.set_format("torch")
            i = torch.tensor(range(len(testset_tokz["input_ids"])))
            test_tokz_iter = TensorDataset(i, testset_tokz["input_ids"])
            test_tokz_iter = DataLoader(
                test_tokz_iter,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                sampler=DistributedSampler(test_tokz_iter, shuffle=False),
            )

            return test_tokz_iter
