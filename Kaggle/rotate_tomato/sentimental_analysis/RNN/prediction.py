import torch

class Prediction():
    def __init__(self):
        pass

    def BiRNN_preds(self, model, test_iter, device):

        model.eval()
        labels=[]
        with torch.no_grad():
            for i, in_id in test_iter:
                in_id=in_id.to(device)
                preds_logit = model(in_id)

                label = torch.argmax(preds_logit, dim=-1).to('cpu').detach().numpy()  # .detach().numpy()的组合用于输出从PyTorch张量转换为NumPy数组
                labels.extend(label)
        return labels
