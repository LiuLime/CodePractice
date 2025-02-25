from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os


def ddp_setup():

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Train:
    def __init__(
        self,
        config,
        train_iter: DataLoader,
        eval_iter: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
    ) -> None:
        self.config = config
        self.epochs = config.num_epochs
        self.batch_size = config.train_batch_size
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.lr = config.lr
        self.optimizer = optimizer
        self.criterion = criterion
        self.start_epoch = 0
        self.snapshot_path = config.model_path
        self.save_every_n_epoch = config.save_every_n_epoch
        self.model = model.to(self.gpu_id)
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()
        self.model = DDP(model, device_ids=[self.gpu_id])
    
    
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
        # print(
        #     f"[GPU{self.gpu_id}] Epoch {epoch+1} | Batchsize: {self.batch_size} | Steps: {len(self.train_iter)}"
        # )
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.train_iter)}"
        )  # test example
        self.train_iter.sampler.set_epoch(epoch)
        for source, targets in self.train_iter:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        ckp = {
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckp, self.snapshot_path)
        # print(f"Epoch {epoch+1} | Training checkpoint saved at {self.snapshot_path}")
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}") # test example


    def _load_snapshot(self):
        loc = f"cuda:{self.gpu_id}"
        ckp = torch.load(self.snapshot_path, map_location=loc)
        self.model.load_state_dict(ckp["model_state_dict"])
        self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
        # start_epoch = ckp["epoch"] + 1
        # print(f"Resuming training from snapshot at Epoch {ckp['epoch']}")

        self.start_epoch = ckp["epoch"] # test example
        print(f"Resuming training from snapshot at Epoch {self.start_epoch}")

        

    # def _evaluate(self, epoch):
    #     eval_loss = 0
    #     correct = 0

    #     with torch.no_grad():
    #         for source, targets in self.eval_iter:
    #             source = source.to(self.gpu_id)
    #             targets = targets.to(self.gpu_id)
    #             outputs = self.model(source)
    #             loss = self.criterion(outputs, targets)

    #             eval_loss += loss.item()
    #             _, pred = outputs.max(1)
    #             correct += (pred == targets).sum().item()

    #     val_avg_loss = eval_loss / len(self.eval_iter)
    #     val_accuracy = 100.0 * correct / len(self.eval_iter.dataset)
        
    #     print(
    #         f"Epoch {epoch+1}/{self.epochs} | Validation avg_loss: {val_avg_loss} | Validation accuracy: {val_accuracy}"
    #     )

    def train(self):
        self._setup_lr_schedule()

        for epoch in range(self.start_epoch, self.epochs):
            self.model.module.train()
            self._run_epoches(epoch)
            
            if self.gpu_id == 0 and epoch % self.save_every_n_epoch == 0:
                   
                self._save_snapshot(epoch)

