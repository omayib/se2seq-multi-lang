from abc import ABC

from torch.utils.data import DataLoader
from tqdm import trange


class Trainer(ABC):
  def __init__(self, train_dataloader:DataLoader, num_epochs:int, learning_rate:float, print_log_frequency:int=10):
    self.train_dataloader = train_dataloader
    self.num_epochs = num_epochs
    self.learning_rate = learning_rate
    self.print_log_frequency=print_log_frequency

    self.num_batches = len(self.train_dataloader)

    if self.num_batches == 0:
      raise ValueError("Empty dataloader. Cannot train without any batches")

  def _train_per_epoch(self)->float:
    raise NotImplemented("Abstract method _train_per_epoch must be implemented.")

  def train(self)->list[float]:
    return [self._train_per_epoch() for _ in trange(self.num_epochs)]
