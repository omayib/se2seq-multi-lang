from typing import Generator

import torch
from torch.utils.data import DataLoader
from torchmetrics.text import ROUGEScore, BLEUScore

from src.inference.predictor import Predictor


class Evaluator:
  def __init__(self, valid_dataloader:DataLoader, predictor:Predictor)-> None:
    self.valid_dataloader = valid_dataloader
    self.predictor=predictor
    # Initialize metric instances once in the constructor
    self._rouge_metric = ROUGEScore(rouge_keys=("rouge1",))
    self._bleu_metric = BLEUScore()

  def _calculate_metrics_per_pair(self, predicted_ids:torch.Tensor, target_ids:torch.Tensor)->Generator[dict, None, None]:
    converter= lambda tensor: " ".join([str(id.item()) for id in tensor])
    for predict_ids, target_ids_single in zip(predicted_ids, target_ids):
      if predict_ids.dim() == 0:
          predict_ids = predict_ids.unsqueeze(0)
      if target_ids_single.dim() == 0:
          target_ids_single = target_ids_single.unsqueeze(0)

      nonzero_index = target_ids_single.count_nonzero().item()

      # Take only relevant tokens (up to EOS or actual length) and convert to string
      predicts = converter(predict_ids[:nonzero_index])
      targets = converter(target_ids_single[:nonzero_index])

      # Calculate ROUGE-1 F1
      rouge_score = self._rouge_metric(predicts, targets)

      # Calculate BLEU score
      # BLEU expects references as a list of list of tokens, or a list of list of strings for string input
      bleu_score = self._bleu_metric(predicts, [[targets]]) # [[targets]] because reference can be multiple for one hypothesis

      # Yield a dictionary of scores
      yield {
          'rouge1_precision': rouge_score["rouge1_precision"].item(),
          'rouge1_recall': rouge_score["rouge1_recall"].item(),
          'rouge1_fmeasure': rouge_score["rouge1_fmeasure"].item(),
          'bleu': bleu_score.item()
      }

  @staticmethod
  def _calculate_accuracy(predicted:torch.Tensor, target:torch.Tensor)->Generator[torch.Tensor, None, None]:
    for predict_ids, target_ids in zip(predicted, target):
      if predict_ids.dim() == 0:
          predict_ids = predict_ids.unsqueeze(0)
      if target_ids.dim() == 0:
          target_ids = target_ids.unsqueeze(0)

      nonzero_index = target_ids.count_nonzero().item()
      print(f"nonzero index {nonzero_index}")
      yield torch.equal(predict_ids[:nonzero_index], target_ids[:nonzero_index])

  @staticmethod
  def _calculate_accuracy_token(predicted:torch.Tensor, target:torch.Tensor)->Generator[torch.Tensor, None, None]:
    for predict_ids, target_ids in zip(predicted, target):
        if predict_ids.dim() == 0:
            predict_ids = predict_ids.unsqueeze(0)
        if target_ids.dim() == 0:
            target_ids = target_ids.unsqueeze(0)

        nonzero_index = target_ids.count_nonzero().item()
        # Compare tokens up to the effective length
        correct_tokens = (predict_ids[:nonzero_index] == target_ids[:nonzero_index]).sum().item()
        total_tokens = nonzero_index # Number of actual tokens in the target

        # Yield the proportion of correct tokens for this sequence
        yield correct_tokens / total_tokens if total_tokens > 0 else 0.0

  def _calculate_average_metric(self, metric_key:str)-> float:
    total_score = 0.0
    num_pairs = 0
    for input_batch, target_batch in self.valid_dataloader:
        input_batch = input_batch.to(self.predictor.device) # Explicitly move input batch to device
        target_batch = target_batch.to(self.predictor.device) # Move target to the same device as predictor
        predicted_batch = self.predictor.predict_by_index(input_batch)
        # Reset metric states for each batch before calculating scores per pair
        self._rouge_metric.reset()
        self._bleu_metric.reset()
        for scores_dict in self._calculate_metrics_per_pair(predicted_batch, target_batch):
            total_score += scores_dict[metric_key]
            num_pairs += 1
    return total_score / num_pairs if num_pairs > 0 else 0.0

  @property
  def rouge1_precision(self)->float:
    return self._calculate_average_metric("rouge1_precision")

  @property
  def rouge1_recall(self)->float:
    return self._calculate_average_metric("rouge1_recall")

  @property
  def rouge1_f1(self)->float:
    return self._calculate_average_metric("rouge1_fmeasure")

  @property
  def bleu_score(self)->float:
    return self._calculate_average_metric("bleu")

  @property
  def accuracy(self)->float:
    accuracy=0
    print(f"evaluator accuracy len : {len(self.valid_dataloader)}")
    for input_batch, target_batch in self.valid_dataloader:
      input_batch = input_batch.to(self.predictor.device) # Explicitly move input batch to device
      target_batch = target_batch.to(self.predictor.device) # Move target to the same device as predictor
      print(f"evaluator accuracy 11 : {accuracy}")
      predicted_batch = self.predictor.predict_by_index(input_batch)
      pridct_acc = self._calculate_accuracy(predicted_batch, target_batch)
      print(f"pridct_acc {sum(pridct_acc)}")
      accuracy += sum(pridct_acc)/len(target_batch)
    return accuracy/len(self.valid_dataloader)

  @property
  def accuracy_by_token(self)->float:
    total_accuracy = 0.0
    count = 0
    for input_batch, target_batch in self.valid_dataloader:
        input_batch = input_batch.to(self.predictor.device) # Explicitly move input batch to device
        target_batch = target_batch.to(self.predictor.device) # Move target to the same device as predictor
        predicted_batch = self.predictor.predict_by_index(input_batch)
        for single_seq_accuracy in self._calculate_accuracy_token(predicted_batch, target_batch):
            total_accuracy += single_seq_accuracy
            count += 1
    return total_accuracy / count if count > 0 else 0.0