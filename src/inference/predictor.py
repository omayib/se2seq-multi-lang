import torch

from src.config import EOS_TOKEN
from src.data.language_data import LanguageData
from src.data.utils import index_tensor_from_sentence
from src.layers.decoder_rnn import DecoderRNN
from src.layers.encoder_rnn import EncoderRNN


class Predictor:
  def __init__(self, encoder: EncoderRNN, decoder:DecoderRNN, input_language:LanguageData, output_language:LanguageData)->None:
    self.encoder = encoder
    self.decoder = decoder
    self.input_language = input_language
    self.output_language = output_language
    self.device = next(encoder.parameters()).device # Get device from encoder

  def predict_by_index(self, input_indexes:torch.Tensor)->torch.Tensor:
    with torch.no_grad():
      input_indexes = input_indexes.to(self.device) # Ensure input is on device
      encoder_outputs, encoder_hidden = self.encoder(input_indexes)
      decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs,encoder_hidden)
      _, selected_ids=decoder_outputs.topk(1)
      decoded_ids=selected_ids.squeeze()
    return decoded_ids

  def predict_by_sentence(self, sentence:str)->torch.Tensor:
    input_indexes = index_tensor_from_sentence(self.input_language, sentence, device=self.device)
    return self.predict_by_index(input_indexes)

  def translate(self, sentence:str)->list[str]:
    decoded_words=[]
    for index in self.predict_by_sentence(sentence):
      if index.item() == EOS_TOKEN:
        break
      decoded_words.append(self.output_language.index_to_word[index.item()])
    return decoded_words

  def get_logged_logits(self):
    """
    Retrieves the logged logits from the decoder.
    Assumes the decoder was initialized with `log_logits_tokens > 0`.
    """
    return self.decoder.logged_logits

print("Predictor class updated with get_logged_logits method.")