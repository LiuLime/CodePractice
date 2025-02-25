from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Train:
    def __init__(
        self,
        config,
        train_iter: DataLoader,
        eval_iter: DataLoader,
        rank: int,
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
        self.model_path = config.model_path
        self.save_every_n_epoch = config.save_every_n_epoch

    def _setup_lr_schedule(self):
        num_training_steps = self.epochs * len(self.train_iter)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def _run_epoches(self, epoch):
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch+1} | Batchsize: {self.batch_size} | Steps: {len(self.train_iter)}"
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
        torch.save(ckp, self.model_path)
        print(f"Epoch {epoch+1} | Training checkpoint saved at {self.model_path}")

    def _load_checkpoint(self):
        ckp = torch.load(self.model_path)
        self.model.module.load_state_dict(ckp["model_state_dict"])
        self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
        start_epoch = ckp["epoch"] + 1
        return start_epoch

    def _evaluate(self, epoch):
        eval_loss = 0
        correct = 0

        with torch.no_grad():
            for source, targets in self.eval_iter:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                outputs = self.model(source)
                loss = self.criterion(outputs, targets)

                eval_loss += loss.item()
                _, pred = outputs.max(1)
                correct += (pred == targets).sum().item()

        val_avg_loss = eval_loss / len(self.eval_iter)
        val_accuracy = 100.0 * correct / len(self.eval_iter.dataset)
        
        print(
            f"Epoch {epoch+1}/{self.epochs} | Validation avg_loss: {val_avg_loss} | Validation accuracy: {val_accuracy}"
        )

    def train(self):
        self._setup_lr_schedule()
        start_epoch = 0
        if self.run_from_ckp is True:
            start_epoch = self._load_checkpoint()

        for epoch in range(start_epoch, self.epochs):
            self.model.module.train()
            self._run_epoches(epoch)
            
            if self.gpu_id == 0:
                self.model.module.eval()
                self._evaluate(epoch)
                if (epoch + 1) % self.save_every_n_epoch == 0:
                    dist.barrier()
                    self._save_checkpoint(epoch)

