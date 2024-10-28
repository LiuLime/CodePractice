import torch
import torch.nn as nn
import torch.nn.functional as F
from dpm.config_ import DLParamConfig
from dpm import config_


class fiveCNN(nn.Module):
    def __init__(self, config: DLParamConfig,
                 num_descriptor_features: int,
                 num_metadata_x_features: int,
                 num_metadata_y_features: int,
                 num_rt_x_features: int):
        super(fiveCNN, self).__init__()
        config_dl = config.cnn_params
        self.lr = config_dl["learning_rate"]
        filters = config_dl["filters"]
        kernel_size = config_dl["kernel_size"]
        strides = config_dl["strides"]
        output_dim = config_dl["output_dim"]
        pool_size = config_dl["pool_size"]
        pool_strides = config_dl["pool_strides"]
        descriptor_dense = 128
        metadata_x_dense = 128
        metadata_y_dense = 128
        # Precursor features - one fully connected layers
        self.descriptor_dense128 = nn.Linear(num_descriptor_features, descriptor_dense)
        # metadata features - one dense layers
        self.metadata_x_dense128 = nn.Linear(num_metadata_x_features, metadata_x_dense)
        self.metadata_y_dense128 = nn.Linear(num_metadata_y_features, metadata_y_dense)

        # Fragment features - convolutional and pooling layers

        self.fragment_conv1 = nn.Conv1d(1, filters, kernel_size, stride=strides)
        self.fragment_conv2 = nn.Conv1d(filters, filters, kernel_size, stride=strides)
        self.fragment_pool1 = nn.MaxPool1d(pool_size, stride=pool_strides)

        self.fragment_conv3 = nn.Conv1d(filters, filters * 2, kernel_size, stride=strides)
        self.fragment_conv4 = nn.Conv1d(filters * 2, filters * 2, kernel_size, stride=strides)
        self.fragment_pool2 = nn.MaxPool1d(pool_size, stride=pool_strides)

        self.fragment_conv5 = nn.Conv1d(filters * 2, filters * 4, kernel_size, stride=strides)
        self.fragment_conv6 = nn.Conv1d(filters * 4, filters * 4, kernel_size, stride=strides)
        self.fragment_conv7 = nn.Conv1d(filters * 4, filters * 4, kernel_size, stride=strides)
        self.fragment_pool3 = nn.MaxPool1d(pool_size, stride=pool_strides)

        self.fragment_conv8 = nn.Conv1d(filters * 4, filters * 8, kernel_size, stride=strides)
        self.fragment_conv9 = nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides)
        self.fragment_conv10 = nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides)
        self.fragment_pool4 = nn.MaxPool1d(pool_size, stride=pool_strides)

        self.fragment_conv11 = nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides)
        self.fragment_conv12 = nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides)
        self.fragment_conv13 = nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides)
        self.fragment_pool5 = nn.MaxPool1d(pool_size, stride=pool_strides)

        self.fragment_flatten = nn.Flatten()

        # Final fully connected layer after concatenation

        # concat_size = 1920
        concat_size = (num_rt_x_features + descriptor_dense + metadata_x_dense + metadata_y_dense)  # 411
        fragment_flatten_size = config.calc_concat_dim(input_dim=concat_size) * filters * 8  # 8*30*8=1920
        self.output_layer = nn.Linear(fragment_flatten_size, output_dim)
        self.init_weights()

    def forward(self, descriptor_features: torch.tensor,
                metadata_x_features: torch.tensor,
                metadata_y_features: torch.tensor,
                rt_x_features: torch.tensor,
                **kwargs):
        # Precursor features processing

        descriptor_dense = F.selu(self.descriptor_dense128(descriptor_features))
        # metadata features - one dense layers
        metadata_x_dense = F.selu(self.metadata_x_dense128(metadata_x_features))
        metadata_y_dense = F.selu(self.metadata_y_dense128(metadata_y_features))

        # print("descriptor feature shape", descriptor_features.shape)
        # print("descriptor dense shape", descriptor_dense.shape)
        # print("rt x features shape", rt_x_features.shape)

        concate_out = torch.cat([descriptor_dense, metadata_x_dense, metadata_y_dense, rt_x_features], dim=-1)
        # print("concated shape", concate_out.shape)

        # Fragment features processing
        fragment_input = concate_out.unsqueeze(1)  # Reshape [batch_size, num_fragment_features] to [batch_size, 1, num_fragment_features] for Conv1d
        # print("block1 fragment_input", fragment_input.shape)
        fragment_out = F.selu(self.fragment_conv1(fragment_input))
        # print("block1 conv1_fragment_out", fragment_out.shape)

        fragment_out = F.selu(self.fragment_conv2(fragment_out))
        # print("block1 conv2_fragment_out", fragment_out.shape)

        fragment_out = self.fragment_pool1(fragment_out)
        # print("block1 maxpool_fragment_out", fragment_out.shape)

        fragment_out = F.selu(self.fragment_conv3(fragment_out))
        fragment_out = F.selu(self.fragment_conv4(fragment_out))
        fragment_out = self.fragment_pool2(fragment_out)

        fragment_out = F.selu(self.fragment_conv5(fragment_out))
        fragment_out = F.selu(self.fragment_conv6(fragment_out))
        fragment_out = F.selu(self.fragment_conv7(fragment_out))
        fragment_out = self.fragment_pool3(fragment_out)

        fragment_out = F.selu(self.fragment_conv8(fragment_out))
        fragment_out = F.selu(self.fragment_conv9(fragment_out))
        fragment_out = F.selu(self.fragment_conv10(fragment_out))
        fragment_out = self.fragment_pool4(fragment_out)

        fragment_out = F.selu(self.fragment_conv11(fragment_out))
        fragment_out = F.selu(self.fragment_conv12(fragment_out))
        fragment_out = F.selu(self.fragment_conv13(fragment_out))
        fragment_out = self.fragment_pool5(fragment_out)

        fragment_out = self.fragment_flatten(fragment_out)

        # Final output_ML layer
        output = F.selu(self.output_layer(fragment_out))  # torch.Size([batch_size,1])
        # print("nnmodel output_ML shape", output_ML.shape)
        return output

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # check bias
                    nn.init.zeros_(m.bias)  # initial bias as 0

        self.apply(_init_weights)


def output_model_params(model: nn.Module):
    for name, param in model.named_parameters():
        print(
            f"Layer: {name} | Size: {param.size()}")


def rmse_loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# #
# if __name__ == '__main__':
#     # Dummy config with embedding size
#     config = config_.DLParamConfig()
#     num_descriptor_features = 286
#     num_metadata_x_features = 282
#     num_metadata_y_features = 282
#     num_rt_x_features = 27
#     # Create the model
#     model = cnnModel(config, num_descriptor_features, num_metadata_x_features, num_metadata_y_features,
#                      num_rt_x_features)
#
#     # Create dummy inputs
#     descriptor_features = torch.randn(10, num_descriptor_features)
#     metadata_x_features = torch.randn(10, num_metadata_x_features)
#     metadata_y_features = torch.randn(10, num_metadata_y_features)
#     rt_x_features = torch.randn(10, num_rt_x_features)
#     # Forward pass
#     output_ML = model(descriptor_features, metadata_x_features, metadata_y_features, rt_x_features)
#     output_model_params(model)
