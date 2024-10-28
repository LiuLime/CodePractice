import torch
import torch.nn as nn
import torch.nn.functional as F
from dpm.config_ import DLParamConfig
from dpm import config_


class fiveCNNwithDropout2(nn.Module):
    def __init__(self, config: DLParamConfig,
                 num_descriptor_features: int,
                 num_metadata_x_features: int,
                 num_metadata_y_features: int,
                 num_rt_x_features: int,
                 dropout: float, ):
        super(fiveCNNwithDropout2, self).__init__()
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
        self.dropout = nn.Dropout(p=dropout)
        # self.relu = F.relu()
        # self.leakyrelu=F.leakyrelu()
        # self.init_weights()

    def forward(self, descriptor_features: torch.tensor,
                metadata_x_features: torch.tensor,
                metadata_y_features: torch.tensor,
                rt_x_features: torch.tensor,
                **kwargs):
        """when mode == train, dropout layer will open, else will close."""
        # # descriptor features processing
        # descriptor_dense = F.relu(self.descriptor_dense128(descriptor_features))
        # # metadata features - one dense layers
        # metadata_x_dense = F.relu(self.metadata_x_dense128(metadata_x_features))
        # metadata_y_dense = F.relu(self.metadata_y_dense128(metadata_y_features))

        # descriptor features processing
        descriptor_dense = F.leaky_relu(self.descriptor_dense128(descriptor_features))
        # metadata features - one dense layers
        metadata_x_dense = F.leaky_relu(self.metadata_x_dense128(metadata_x_features))
        metadata_y_dense = F.leaky_relu(self.metadata_y_dense128(metadata_y_features))

        # Model v3.
        descriptor_dense = self.dropout(descriptor_dense)
        metadata_x_dense = self.dropout(metadata_x_dense)
        metadata_y_dense = self.dropout(metadata_y_dense)

        # concat features
        concate_out = torch.cat([descriptor_dense, metadata_x_dense, metadata_y_dense, rt_x_features], dim=-1)

        # Fragment features processing
        fragment_input = concate_out.unsqueeze(
            1)  # Reshape (batch_size, num_fragment_features) -> (batch_size, 1, num_fragment_features) for Conv1d
        # print("block1 fragment_input", fragment_input.shape)
        # fragment_out = F.relu(self.fragment_conv1(fragment_input))
        # fragment_out = F.relu(self.fragment_conv2(fragment_out))
        # fragment_out = self.fragment_pool1(fragment_out)

        # fragment_out = F.relu(self.fragment_conv3(fragment_out))
        # fragment_out = F.relu(self.fragment_conv4(fragment_out))
        # fragment_out = self.fragment_pool2(fragment_out)

        # fragment_out = F.relu(self.fragment_conv5(fragment_out))
        # fragment_out = F.relu(self.fragment_conv6(fragment_out))
        # fragment_out = F.relu(self.fragment_conv7(fragment_out))
        # fragment_out = self.fragment_pool3(fragment_out)

        # fragment_out = F.relu(self.fragment_conv8(fragment_out))
        # fragment_out = F.relu(self.fragment_conv9(fragment_out))
        # fragment_out = F.relu(self.fragment_conv10(fragment_out))
        # fragment_out = self.fragment_pool4(fragment_out)

        # fragment_out = F.relu(self.fragment_conv11(fragment_out))
        # fragment_out = F.relu(self.fragment_conv12(fragment_out))
        # fragment_out = F.relu(self.fragment_conv13(fragment_out))
        fragment_out = F.leaky_relu(self.fragment_conv1(fragment_input))
        fragment_out = F.leaky_relu(self.fragment_conv2(fragment_out))
        fragment_out = self.fragment_pool1(fragment_out)

        fragment_out = F.leaky_relu(self.fragment_conv3(fragment_out))
        fragment_out = F.leaky_relu(self.fragment_conv4(fragment_out))
        fragment_out = self.fragment_pool2(fragment_out)

        fragment_out = F.leaky_relu(self.fragment_conv5(fragment_out))
        fragment_out = F.leaky_relu(self.fragment_conv6(fragment_out))
        fragment_out = F.leaky_relu(self.fragment_conv7(fragment_out))
        fragment_out = self.fragment_pool3(fragment_out)

        fragment_out = F.leaky_relu(self.fragment_conv8(fragment_out))
        fragment_out = F.leaky_relu(self.fragment_conv9(fragment_out))
        fragment_out = F.leaky_relu(self.fragment_conv10(fragment_out))
        fragment_out = self.fragment_pool4(fragment_out)

        fragment_out = F.leaky_relu(self.fragment_conv11(fragment_out))
        fragment_out = F.leaky_relu(self.fragment_conv12(fragment_out))
        fragment_out = F.leaky_relu(self.fragment_conv13(fragment_out))
        fragment_out = self.fragment_pool5(fragment_out)

        fragment_out = self.fragment_flatten(fragment_out)

        # Final output layer
        output = F.relu(self.output_layer(fragment_out))  # torch.Size([batch_size,1])
        return output

    # def init_weights(self):
    #     def _init_weights(m):
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.zeros_(m.bias)

    #     self.apply(_init_weights)



def output_model_params(model: nn.Module):
    for name, param in model.named_parameters():
        print(
            f"Layer: {name} | Size: {param.size()}")


def rmse_loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))
