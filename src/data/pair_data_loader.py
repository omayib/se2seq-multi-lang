from torch.utils.data import DataLoader

from src.data.language_data import LanguageData
from src.data.pair_dataset import PairDataset


class PairDataLoader:
  def __init__(self,pairs:list[list[str]],input_language=LanguageData, output_language=LanguageData, batch_size=int, num_workers:int = 2):
    first_five = pairs[:5]
    last_five = pairs[-5:]
    print(f"PairDataLoader input {input_language.get_name()} ,  output : {output_language.get_name()}")
    print(f"PairDataLoader first_five ::  {first_five}")
    print(f"PairDataLoader last_five ::  {last_five}")
    self.dataset = PairDataset(pairs, input_language, output_language)
    self.batch_size = batch_size
    self.num_workers = num_workers

  @property
  def load(self)->DataLoader:
    return DataLoader(self.dataset(), shuffle=False,batch_size=self.batch_size,num_workers=self.num_workers)

  @property
  def validation_dataloader(self)-> DataLoader:
    return DataLoader(self.dataset(), shuffle=False,batch_size=self.batch_size,num_workers=self.num_workers)