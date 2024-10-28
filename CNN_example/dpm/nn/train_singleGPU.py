from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch

from tqdm.auto import tqdm
import datetime
import os
import sys

sys.path.append("../..")

from dpm.utils import log, common
from dpm.nn.early_stopping import EarlyStopping

logger = log.logger()


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            eval_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion,
            device,
            epochs: int,
            save_every: int,
            model_path: str,
            metric_path: str,
            run_from_ckp: bool,
            patience: int | None,
    ) -> None:
        self.gpu_id = device
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.eval_iter = eval_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.save_every = save_every
        self.num_training_steps = self.epochs * len(self.train_data)
        self.model_path = model_path
        self.metric_path = metric_path
        self.patience = patience

        # self.progress_bar = tqdm(range(self.num_training_steps))

        if run_from_ckp is True:
            self.start_epoch = self._load_checkpoint()
            self.metric = common.read_json(self.metric_path)
        else:
            self.start_epoch = 0
            self.metric = []

    def _run_batch(self, descriptor_features, metadata_x_features, metadata_y_features, rt_x_features, targets):
        self.optimizer.zero_grad()
        self.model.train()  # dropout enable
        output = self.model(descriptor_features, metadata_x_features, metadata_y_features, rt_x_features)
        # print("train_singleGPU: _run_batch 测试|targets shape", targets.shape)
        # print("train_singleGPU: _run_batch 测试|output_ML shape", output_ML.shape)
        # print("train_singleGPU: _run_batch 测试|descriptor_features shape", descriptor_features.shape)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        # self.progress_bar.update(1)
        # Calculate train loss without dropout
        train_loss_without_dropout = self._train_loss_without_dropout(
            descriptor_features, metadata_x_features, metadata_y_features, rt_x_features, targets
        )
        return train_loss_without_dropout

    def _train_loss_without_dropout(self, descriptor_features, metadata_x_features, metadata_y_features, rt_x_features,
                                    targets):
        """Calculate train loss without dropout layer for model with dropout"""
        # Switch to evaluation mode to disable dropout
        self.model.eval()
        with torch.no_grad():  # Disable gradient calculation for efficiency
            output = self.model(descriptor_features, metadata_x_features, metadata_y_features, rt_x_features)
            loss = self.criterion(output, targets)
        # Switch back to training mode for further training
        self.model.train()
        return loss.item()

    def _run_epoch(self, epoch):
        train_loss = 0
        b_sz = len(next(iter(self.train_data))[0])
        logger.debug(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        # self.train_data.sampler.set_epoch(epoch)
        for descriptor_features, metadata_x_features, metadata_y_features, rt_x_features, targets in self.train_data:
            descriptor_features = descriptor_features.to(self.gpu_id)
            metadata_x_features = metadata_x_features.to(self.gpu_id)
            metadata_y_features = metadata_y_features.to(self.gpu_id)
            rt_x_features = rt_x_features.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            train_loss += self._run_batch(descriptor_features, metadata_x_features, metadata_y_features, rt_x_features,
                                          targets)
        train_loss_ave = train_loss / len(self.train_data)
        return train_loss_ave

    def _save_checkpoint(self, epoch):
        """save model state dict, optimizer state dict and epoch info, deal with single gpu/cpu model object and DataParallel/DDP wrapped model object"""
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        ckp = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        PATH = self.model_path
        torch.save(ckp, PATH)
        logger.debug(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _load_checkpoint(self):
        ckp = torch.load(self.model_path)
        # self.model.module.load_state_dict(ckp["model_state_dict"])  # 分布式训练用法
        self.model.load_state_dict(ckp['model_state_dict'])  # 如果不是分布式训练（但是chatgpt说分布式也用这个model.load_state_dict）
        self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
        start_epoch = ckp["epoch"] + 1
        return start_epoch

    def _setup_lr_schedule(self):
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

    def _save_metric(self, epoch, train_avg_loss, val_avg_loss):
        temp_metric = {"epoch": epoch,
                       "train_average_loss": train_avg_loss,
                       "val_avg_loss": val_avg_loss
                       }
        self.metric.append(temp_metric)
        common.save_json(self.metric, self.metric_path)

    def _evaluate(self):
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for descriptor_features, metadata_x_features, metadata_y_features, rt_x_features, targets in self.eval_iter:
                descriptor_features = descriptor_features.to(self.gpu_id)
                metadata_x_features = metadata_x_features.to(self.gpu_id)
                metadata_y_features = metadata_y_features.to(self.gpu_id)
                rt_x_features = rt_x_features.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                outputs = self.model(descriptor_features, metadata_x_features, metadata_y_features, rt_x_features)
                # print(f"epoch {epoch} outputs:",outputs.shape)  # tensor[[xx],[xx],[xx]] torch.Size([64,1,1]) or torch.Size([22,1,1])
                loss = self.criterion(outputs, targets)
                eval_loss += loss.item()

        val_avg_loss = eval_loss / len(self.eval_iter)  # len(self.eval_iter)=3

        return val_avg_loss

    def train(self):
        """train func without early stopping"""
        self._setup_lr_schedule()

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            train_avg_loss = self._run_epoch(epoch)
            if (epoch + 1) % self.save_every == 0:
                self.model.eval()
                val_avg_loss = self._evaluate()
                logger.debug(
                    f"Epoch {epoch}/{self.epochs - 1}| train avg_loss:{train_avg_loss} | Validation avg_loss: {val_avg_loss}")

                self._save_metric(epoch, train_avg_loss, val_avg_loss)
                self._save_checkpoint(epoch)
        # 确保最后一次保存和评价了
        if self.epochs % self.save_every != 0:
            self.model.eval()
            val_avg_loss = self._evaluate()
            logger.debug(
                f"Epoch {self.epochs - 1}/{self.epochs - 1}| train avg_loss:{train_avg_loss} | Validation avg_loss: {val_avg_loss}")
            self._save_metric(self.epochs - 1, train_avg_loss, val_avg_loss)
        self._save_checkpoint(self.epochs - 1)

    def train_with_earlystop(self):
        """train func with early stopping, save every epoch"""
        self._setup_lr_schedule()
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=0, path=self.model_path,
                                       trace_func=logger.debug)
        for epoch in range(self.start_epoch, self.epochs):
            train_avg_loss = self._run_epoch(epoch)
            val_avg_loss = self._evaluate()
            self._save_metric(epoch, train_avg_loss, val_avg_loss)
            logger.debug(
                f"Epoch {epoch}/{self.epochs - 1}| train avg_loss:{train_avg_loss} | Validation avg_loss: {val_avg_loss}")

            if early_stopping(val_avg_loss) is True:
                self._save_checkpoint(epoch)
            if early_stopping.early_stop:
                logger.debug("Early stopping triggered. Ending training.")
                break


def custom_prepare_batch(batch, device=None, non_blocking=False):
    descriptor_features, metadata_x_features, metadata_y_features, rt_x_features, labels = batch
    descriptor_features = descriptor_features.to(device, non_blocking=non_blocking)
    metadata_x_features = metadata_x_features.to(device, non_blocking=non_blocking)
    metadata_y_features = metadata_y_features.to(device, non_blocking=non_blocking)
    rt_x_features = rt_x_features.to(device, non_blocking=non_blocking)
    labels = labels.to(device, non_blocking=non_blocking)
    
    # 返回所有特征和标签作为元组，确保特征顺序与模型 forward 的输入一致
    return (descriptor_features, metadata_x_features, metadata_y_features, rt_x_features), labels


def lr_finder(model, criterion, optimizer, train_loader, eval_loader,device, start_lr:float =1e-7,end_lr: float = 10.0, num_iter: int = 100,  step_mode:str='exp', diverge_th=1.5, save_path="lr_finder_plot.pdf") -> None:
    from matplotlib import pyplot as plt
    from ignite.handlers import FastaiLRFinder
    from ignite.engine import create_supervised_trainer, create_supervised_evaluator
    from ignite.metrics import Metric, Loss, Accuracy,MeanSquaredError,MeanAbsoluteError
    import torch.nn as nn
    
    logger.debug("processing learning rate finder +++++++++++")
    # save the initial state for the model and the optimizer
    # to reset them later
    init_model_state = model.state_dict()
    init_opt_state = optimizer.state_dict()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device, prepare_batch=custom_prepare_batch)

    lr_finder = FastaiLRFinder()
    to_save={'model': model, 'optimizer': optimizer}
    with lr_finder.attach(trainer, to_save, 
    num_iter=num_iter, 
    start_lr=start_lr, 
    end_lr=end_lr, 
    step_mode=step_mode,
    diverge_th=diverge_th) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(train_loader)
    ax = lr_finder.plot()
    plt.savefig(save_path)
    plt.close()

    print("Suggested LR", lr_finder.lr_suggestion())
  

    # with lr=3e-4
    trainer.run(train_loader, max_epochs=5)

    evaluator = create_supervised_evaluator(model, 
    metrics={"mse": MeanSquaredError(), "mae":MeanAbsoluteError(),"loss": Loss(nn.MSELoss())}, 
    device=device, 
    prepare_batch=custom_prepare_batch)
    evaluator.run(eval_loader)
    print("init lr:",optimizer.param_groups[0]['lr'])
    print("init lr eval metric:",evaluator.state.metrics)
    
    # After training we need to get the model and the optimizer back to their initial state, to do that, we will load the initial state again.

    # Reinit model / optimizer
    model.load_state_dict(init_model_state)
    optimizer.load_state_dict(init_opt_state)
    # Let's now apply suggested learning rate to the optimizer, and train the model again with optimal learning rate.

    lr_finder.apply_suggested_lr(optimizer)
    print("suggested lr:",optimizer.param_groups[0]['lr'])
    trainer.run(train_loader, max_epochs=5)

    evaluator.run(eval_loader)
    print("suggested lr eval metric:",evaluator.state.metrics)