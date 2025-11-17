from abc import ABC, abstractmethod

import torch
from torch import nn


class Decoder(ABC, nn.Module):
  def __init__(self, hidden_size:int, output_size:int, dropout_rate:float, device:str)->None:
    super().__init__()

    self.hidden_size = hidden_size
    self.output_size = output_size
    self.dropout_rate = dropout_rate
    self.device = device
    self.embedding = nn.Embedding(self.output_size, self.hidden_size)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
    self.output_layer = nn.Linear(self.hidden_size, self.output_size)
    self.dropout = nn.Dropout(self.dropout_rate)

  @abstractmethod
  def forward(self, encoder_outputs : torch.Tensor,
              encoder_hidden:torch.Tensor,
              target_tensor:torch.Tensor=None)-> tuple[torch.Tensor]:
              return NotImplementedError