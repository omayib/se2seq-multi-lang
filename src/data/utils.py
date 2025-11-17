import re
import unicodedata

import torch

from src.config import EOS_TOKEN
from src.data.language_data import LanguageData

def clean_punctuation(string:str)->str:
    clean_spcl = re.compile('[/(){}\[\]\|@,;]')
    clean_symbol = re.compile('[^0-9a-z]')
    text = clean_spcl.sub('',string)
    text = clean_symbol.sub(' ', text)
    return text

def remove_accents(string: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", string)
        if not unicodedata.combining(char)
    )
def normalize_string(string: str) -> str:
    lower_string = string.lower().strip()
    removed_punctuation = clean_punctuation(lower_string)
    removed_accents = remove_accents(removed_punctuation)
    # target_punctuations = re.sub(r"([.!?])", r" \1", removed_accents)
    # removed_punctuations = re.sub(r"[^a-zA-Z!?]+", r" ", target_punctuations)
    return removed_accents.strip()

def index_from_sentence(language:LanguageData, sentence:str)->list[int]:
  return [language.word_to_index[word] for word in sentence.split(" ")]

def index_tensor_from_sentence(language:LanguageData, sentence:str, device:str="cpu")-> torch.tensor:
  indexes = index_from_sentence(language, sentence)
  indexes.append(EOS_TOKEN)
  return torch.tensor(indexes, dtype=torch.long, device=device).view(1,-1)