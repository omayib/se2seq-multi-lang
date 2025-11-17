import torch
import torch.nn.functional as F

from src.config import SOS_TOKEN, MAX_LENGTH
from src.layers.decoder import Decoder


class DecoderRNN(Decoder):
  def __init__(self, hidden_size:int, output_size:int, dropout_rate:float, device:str, max_length:int = MAX_LENGTH, log_logits_tokens:int=0)->None:
    super().__init__(hidden_size, output_size, dropout_rate, device)
    self.max_length = max_length
    self.log_logits_tokens = log_logits_tokens
    self.logged_logits = None

  def forward(self, encoder_outputs:torch.Tensor,encoder_hidden:torch.Tensor,target_tensor:torch.Tensor=None)->tuple[torch.Tensor]:
    batch_size = encoder_outputs.size(0)
    decoder_input = torch.empty(batch_size,1,dtype=torch.long, device=self.device).fill_(SOS_TOKEN)
    decoder_hidden = encoder_hidden
    decoder_outputs =[]
    all_logits = [] # To store logits for logging

    for i in range(self.max_length):
      decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
      decoder_outputs.append(decoder_output)
      if self.log_logits_tokens > 0 and i < self.log_logits_tokens:
          all_logits.append(decoder_output.squeeze(1))

      if target_tensor is not None:
        decoder_input = target_tensor[:,i].unsqueeze(1)
      else:
        _, top_i = decoder_output.topk(1)
        decoder_input = top_i.squeeze(-1).detach()

    decoder_outputs = torch.cat(decoder_outputs,dim=1)
    decoder_probs = F.log_softmax(decoder_outputs, dim=-1)

    if len(all_logits) > 0:
        self.logged_logits = torch.stack(all_logits, dim=1)

    return decoder_probs, decoder_hidden,None

  def forward_step(self, input:torch.Tensor, hidden = torch.Tensor)-> tuple[torch.Tensor]:
    embedding = self.embedding(input)
    output, hidden = self.gru(F.relu(embedding),hidden)
    logits = self.output_layer(output)
    return logits, hidden

print("DecoderRNN class updated to log initial token logits.")