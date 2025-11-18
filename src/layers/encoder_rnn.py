import torch
from torch import nn


class EncoderRNN(nn.Module):
  def __init__ (self, input_size:int, hidden_size:int, dropout_rate:float,num_layers:int=1)->None:
    super().__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(
        hidden_size,
        hidden_size,
        batch_first=True,
        num_layers=num_layers,  # Pass the new argument here
        dropout=dropout_rate if num_layers > 1 else 0.0  # Standard practice: inter-layer dropout
    )
    self.dropout_rate = dropout_rate
    self.dropout = nn.Dropout(self.dropout_rate)

  def forward(self, _input:torch.Tensor)-> tuple[torch.Tensor]:
    embedding = self.dropout(self.embedding(_input))
    output, hidden = self.gru(embedding)
    hidden = self.dropout(hidden)
    output = self.dropout(output)
    return output, hidden