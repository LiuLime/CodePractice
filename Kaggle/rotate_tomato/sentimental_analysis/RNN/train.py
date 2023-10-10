import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from transformers import get_scheduler
import torch

class Train():
    def __init__(self, config, train_iter, eval_iter):
        self.config = config
        self.epochs = config.num_epochs
        self.lr = config.lr
        self.train_iter = train_iter
        self.device = config.device
        self.eval_iter = eval_iter
        self.save_path= config.model_path

    def BiRNN_train(self, model):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        num_training_steps = self.epochs * len(self.train_iter)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
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
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.epochs,
                                                                             epoch * len(self.train_iter) + i + 1,
                                                                             num_training_steps, loss.item()))
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
            val_accuracy = 100. * correct / len(self.eval_iter.dataset)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Average Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        torch.save(model.state_dict(), f'{self.save_path}/BiRNN_params.pt')

# %%
