import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.config import MAX_LENGTH, PAD_TOKEN, EOS_TOKEN
from src.data.language_data import LanguageData
from src.data.utils import index_from_sentence


class PairDataset:
  def __init__(self, pairs:list[list[str]], input_language:LanguageData, output_language:LanguageData)->None:
    first_five = pairs[:5]
    last_five = pairs[-5:]
    print(f"PairDataset input {input_language.get_name()} ,  output : {output_language.get_name()}")
    print(f"PairDataset first_five ::  {first_five}")
    print(f"PairDataset last_five ::  {last_five}")
    self.pairs = pairs
    self.input_language = input_language
    self.output_language = output_language

  def __len__(self)->int:
    return len(self.pairs)

  def __call__(self)->TensorDataset:
    n = len(self.pairs)
    # Initialize with PAD_TOKEN (2) instead of 0 (SOS_TOKEN)
    input_ids = np.full((n,MAX_LENGTH), PAD_TOKEN, dtype=np.int32)
    target_ids = np.full((n,MAX_LENGTH), PAD_TOKEN, dtype=np.int32)

    for index, (_input,target) in enumerate(self.pairs):
      inputs = index_from_sentence(self.input_language, _input)
      targets = index_from_sentence(self.output_language, target)

      inputs.append(EOS_TOKEN)
      targets.append(EOS_TOKEN)

      input_ids[index, :len(inputs)] = inputs
      target_ids[index, :len(targets)] =  targets
    return TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(target_ids))