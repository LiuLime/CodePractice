import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"  # 表示分布式训练的主节点地址在当前机器
    os.environ["MASTER_PORT"] = "12355"  # 主节点监听的端口号,可以是从1024到65535(0-1023为知名占用端口)
    # 的任意未被占用的端口值

    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Train:
    def __init__(
        self,
        config,
        train_iter: DataLoader,
        eval_iter: DataLoader,
        rank: int,
        save_every_n_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
    ) -> None:
        self.config = config
        self.run_from_ckp = config.run_from_ckp
        self.epochs = config.num_epochs
        self.batch_size = config.train_batch_size
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.gpu_id = rank
        self.model = model.to(rank)
        self.model = DDP(model, device_ids=[rank])
        self.lr = config.lr
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_path = config.model_path
        self.save_step = save_every_n_step

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epches(self, epoch):
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.train_iter)}"
        )
        self.train_iter.sampler.set_epoch(epoch)
        for source, targets in self.train_iter:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = {
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckp, self.save_path)
        print(f"Epoch {epoch+1} | Training checkpoint saved at {self.save_path}")

    def _load_checkpoint(self):
        ckp = torch.load(self.save_path)
        self.model.module.load_state_dict(ckp['model_state_dict'])
        self.optimizer.load_state_dict(ckp['optimizer_state_dict']) 
        start_epoch = ckp['epoch']+1
        return start_epoch
    
    def train(self):
        start_epoch = 0
        if self.run_from_ckp is True:
            start_epoch = self._load_checkpoint()
            
        for epoch in range(start_epoch, self.epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch)

    def BiRNN_train(self, model):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        num_training_steps = self.epochs * len(self.train_iter)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps))

        for epoch in range(self.epochs):
            model.train()
            total_train_loss = 0
            for i, (input_id, label) in enumerate(self.train_iter):
                input_id = input_id.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()

                out = model(input_id)  # 前向传播
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()  # 调整学习率
                total_train_loss += loss.item()
                progress_bar.update(1)

                if (i + 1) % 200 == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch + 1,
                            self.epochs,
                            epoch * len(self.train_iter) + i + 1,
                            num_training_steps,
                            loss.item(),
                        )
                    )
            avg_train_loss = total_train_loss / len(self.train_iter)

            # validation
            model.eval()  # 设置模型为评估模式
            eval_loss = 0
            correct = 0

            with torch.no_grad():
                for inputs, labels in self.eval_iter:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item()

                    _, pred = outputs.max(1)
                    correct += (pred == labels).sum().item()

            avg_val_loss = eval_loss / len(self.eval_iter)
            val_accuracy = 100.0 * correct / len(self.eval_iter.dataset)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Average Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
            )

        torch.save(model.state_dict(), f"{self.save_path}/BiRNN_params.pt")


# %%
