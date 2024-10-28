import torch


class Prediction():
    def __init__(self):
        pass

    # @staticmethod
    # def nn_preds(model, test_iter, device):
    #     model.eval()
    #     labels = []
    #     with torch.no_grad():
    #         for i, in_id in test_iter:
    #             in_id = in_id.to(device)
    #             preds = model(in_id)
    #
    #             label = preds.to('cpu').detach().numpy()  # .detach().numpy()的组合用于输出从PyTorch张量转换为NumPy数组
    #             labels.extend(label)
    #     return labels

    @staticmethod
    def nn_preds(model, test_loader, device):
        print("preds test",device)
        model.eval()
        labels = []
        with torch.no_grad():
            for descriptor_features, metadata_x_features, metadata_y_features, rt_x_features, _ in test_loader:
                descriptor_features = descriptor_features.to(device)
                metadata_x_features = metadata_x_features.to(device)
                metadata_y_features = metadata_y_features.to(device)
                rt_x_features = rt_x_features.to(device)

                outputs = model(descriptor_features, metadata_x_features, metadata_y_features, rt_x_features)
                # print("测试preds_singleGPU｜test set prediction outputs shape:", outputs.shape)
                label = outputs.to('cpu').squeeze(1).detach().numpy()  # .detach().numpy()的组合用于输出从PyTorch张量转换为NumPy数组
                print("测试preds_singleGPU｜test set prediction label shape:", label.shape)
                labels.extend(label)
        return labels
