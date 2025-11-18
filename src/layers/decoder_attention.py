import torch
from torch import nn

from src.config import SOS_TOKEN, MAX_LENGTH
from src.layers.attention import Attention
from src.layers.decoder import Decoder
import torch.nn.functional as F


class AttentionDecoderRNN(Decoder):
  def __init__(self, hidden_size:int, output_size:int, dropout_rate:float, device:str, max_length:int = MAX_LENGTH, log_logits_tokens:int=0,num_layers:int=1)->None:
    super().__init__(hidden_size, output_size, dropout_rate, device)
    self.max_length = max_length
    self.num_layers = num_layers
    self.embedding = nn.Embedding(self.output_size, self.hidden_size)
    self.attention = Attention(self.hidden_size)
    self.gru = nn.GRU(
        2 * self.hidden_size,  # Input size: Embedding + Context Vector
        self.hidden_size,
        batch_first=True,
        num_layers=num_layers,  # Pass the layer count
        dropout=dropout_rate if num_layers > 1 else 0.0  # Built-in inter-layer dropout
    )
    self.log_logits_tokens = log_logits_tokens
    self.logged_logits = None

  def forward(self, encoder_outputs: torch.Tensor, encoder_hidden: torch.Tensor, target_tensor: torch.Tensor = None) -> tuple[torch.Tensor]:
    batch_size = encoder_outputs.size(0)
    decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_TOKEN)
    decoder_hidden = encoder_hidden
    decoder_outputs = []
    attentions =[]
    all_logits = [] # To store logits for logging

    for i in range(self.max_length):
      decoder_output, decoder_hidden, attention = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
      decoder_outputs.append(decoder_output)
      attentions.append(attention)
      if self.log_logits_tokens > 0 and i < self.log_logits_tokens:
          all_logits.append(decoder_output.squeeze(1))

      if target_tensor is not None:
        decoder_input = target_tensor[:, i].unsqueeze(1)
      else:
        _, top_i = decoder_output.topk(1)
        decoder_input = top_i.squeeze(-1).detach()

    decoder_outputs = torch.cat(decoder_outputs, dim=1)
    decoder_probs = F.log_softmax(decoder_outputs, dim=-1)
    attentions = torch.cat(attentions, dim=1)

    if len(all_logits) > 0:
        self.logged_logits = torch.stack(all_logits, dim=1)

    return decoder_probs, decoder_hidden, attentions

  def forward_step(self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> tuple[torch.Tensor]:
    embedding = self.dropout(self.embedding(input))
    if self.num_layers > 1:
        query = hidden[-1, :, :].unsqueeze(0).permute(1, 0, 2)
    else:
        query = hidden.permute(1, 0, 2)
    context, attention = self.attention(query, encoder_outputs)
    input_gru = torch.cat((embedding, context), dim=2)
    output, hidden = self.gru(input_gru, hidden)
    output = self.dropout(output)
    logits = self.output_layer(output)
    return logits, hidden, attention

print("AttentionDecoderRNN class updated to log initial token logits.")