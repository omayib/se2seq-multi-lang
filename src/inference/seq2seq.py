import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import trange

from src.config import SOS_TOKEN, TO_JV_TOKEN, TO_SU_TOKEN, PAD_TOKEN, EOS_TOKEN
from src.data.language_data import LanguageData
from src.inference.evaluator import Evaluator
from src.inference.predictor import Predictor
from src.inference.trainer import Trainer
from src.layers.decoder import Decoder
from src.layers.encoder_rnn import EncoderRNN


class Seq2SeqTrainer(Trainer):
  def __init__(self, train_dataloader:DataLoader, encoder:EncoderRNN, decoder:Decoder, num_epochs:int, learning_rate:float, input_language:LanguageData, output_language:LanguageData, valid_dataloader:DataLoader, print_log_frequency:int=10, debug_log_tokens_every_n_epochs: int = 0, debug_num_logged_tokens: int = 0):
    super().__init__(train_dataloader=train_dataloader, num_epochs=num_epochs,learning_rate=learning_rate,print_log_frequency=print_log_frequency)

    self.encoder=encoder
    self.decoder =decoder
    self.input_language = input_language
    self.output_language = output_language
    self.valid_dataloader = valid_dataloader
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Store device for tensor transfers

    self._criterion = nn.NLLLoss(ignore_index=PAD_TOKEN) # Modified to ignore PAD_TOKEN
    self._encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
    self._decoder_optimizer=optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

    self.epoch_accuracies: list[float] = []
    self.epoch_rouge_f1s: list[float] = []
    self.epoch_bleu_scores: list[float] = []

    # Debugging parameters
    self.debug_log_tokens_every_n_epochs = debug_log_tokens_every_n_epochs
    self.debug_num_logged_tokens = debug_num_logged_tokens

  def _train_per_epoch(self)-> float:
    total_loss=0

    for input_tensor,target_tensor in self.train_dataloader:
      input_tensor = input_tensor.to(self.device) # Move input to device
      target_tensor = target_tensor.to(self.device) # Move target to device

      self._encoder_optimizer.zero_grad()
      self._decoder_optimizer.zero_grad()

      encoder_outputs, encoder_hidden = self.encoder(input_tensor)
      decoder_outputs, _,_ = self.decoder(encoder_outputs, encoder_hidden, target_tensor)
      loss = self._criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)),target_tensor.view(-1))
      loss.backward()
      self._encoder_optimizer.step()
      self._decoder_optimizer.step()

      total_loss += loss.item()
    return total_loss / self.num_batches

  def train(self)-> tuple[list[float], list[float], list[float], list[float]]: # Added list[float] for BLEU
    epoch_losses = []
    for epoch in trange(self.num_epochs):
      epoch_loss = self._train_per_epoch()
      epoch_losses.append(epoch_loss)

      # Evaluation part
      self.encoder.eval()
      self.decoder.eval()
      with torch.no_grad():
        predictor = Predictor(encoder=self.encoder, decoder=self.decoder, input_language=self.input_language, output_language=self.output_language)
        evaluator = Evaluator(valid_dataloader=self.valid_dataloader, predictor=predictor)

        current_accuracy = evaluator.accuracy_by_token
        current_rouge_f1 = evaluator.rouge1_f1
        current_bleu_score = evaluator.bleu_score # Get BLEU score

        self.epoch_accuracies.append(current_accuracy)
        self.epoch_rouge_f1s.append(current_rouge_f1)
        self.epoch_bleu_scores.append(current_bleu_score) # Append BLEU score

        # Debugging: Log initial token logits if enabled
        if self.debug_log_tokens_every_n_epochs > 0 and (epoch + 1) % self.debug_log_tokens_every_n_epochs == 0:
            print(f"\n--- Epoch {epoch+1}: Debugging Logits ---")
            # Temporarily enable logging in decoder
            original_log_logits_tokens = self.decoder.log_logits_tokens
            self.decoder.log_logits_tokens = self.debug_num_logged_tokens

            # Get a sample input tensor from the validation dataloader
            sample_input_batch, _ = next(iter(self.valid_dataloader))
            sample_input = sample_input_batch[0].unsqueeze(0).to(self.device)

            # Make a prediction to trigger logit logging
            _ = predictor.predict_by_index(sample_input)
            logged_logits = predictor.get_logged_logits()

            if logged_logits is not None:
                # Assuming batch_size=1 for debugging single sample
                for i in range(logged_logits.shape[1]):
                    token_logits = logged_logits[0, i, :].cpu().numpy()
                    print(f"  Logits for predicted token {i+1}:")
                    print(f"    SOS_TOKEN (0, 'SOS'): {token_logits[SOS_TOKEN]:.4f}")
                    print(f"    EOS_TOKEN (1, 'EOS'): {token_logits[EOS_TOKEN]:.4f}")
                    print(f"    TO_JV_TOKEN (3, '<to_javanese>'): {token_logits[TO_JV_TOKEN]:.4f}")
                    print(f"    TO_SU_TOKEN (4, '<to_sundanese>'): {token_logits[TO_SU_TOKEN]:.4f}")
                    max_logit_idx = np.argmax(token_logits)
                    max_logit_value = token_logits[max_logit_idx]
                    max_logit_word = self.output_language.index_to_word.get(max_logit_idx, f"<unk_{max_logit_idx}>")
                    print(f"    Highest logit: {max_logit_value:.4f} for token '{max_logit_word}' (index {max_logit_idx})")
            else:
                print("    No logits were logged for this sample.")

            # Reset log_logits_tokens in decoder
            self.decoder.log_logits_tokens = original_log_logits_tokens
            print("------------------------------------")

      self.encoder.train()
      self.decoder.train()

    return epoch_losses, self.epoch_accuracies, self.epoch_rouge_f1s, self.epoch_bleu_scores