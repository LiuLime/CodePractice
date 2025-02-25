import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiRNN(nn.Module):
    def __init__(self, config):
        super(BiRNN, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.output_dim = config.output_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * self.hidden_dim, self.output_dim)
        self.soft = nn.Softmax(dim=1)
        self.init_weights()

    def forward(self, x):  # input (batch_size, seq_length)

        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        self.encoder.flatten_parameters()

        out, _ = self.encoder(embeds)  # out(batch_size, seq_length, 2*hidden_dim) torch.Size([24,512,512])

        first_timestep = out[:, 0, :]  # fist time step[batch_size, 2*hidden_dim]
        last_timestep = out[:, -1, :]  # last time step[batch_size, 2*hidden_dim]
        out2 = torch.cat((first_timestep, last_timestep), dim=1)

        out2 = self.decoder(out2)  # (batch_size, 4 * hidden_dim) -> (batch_size, output_dim)
        soft_out = self.soft(out2)

        return soft_out

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # check bias
                    nn.init.zeros_(m.bias)  # initial bias as 0
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        self.apply(_init_weights)  # apply _init_weights to every submodules
