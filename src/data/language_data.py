from collections import defaultdict

from src.config import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, TO_JV_TOKEN, TO_SU_TOKEN, TO_JV_TOKEN_STR, TO_SU_TOKEN_STR


class LanguageData:
  def __init__(self, name:str)->None:
    self.name = name
    self.word_to_index = {}
    self.word_to_count = defaultdict(int)
    self.index_to_word = {
        SOS_TOKEN:"SOS",
        EOS_TOKEN:"EOS",
        PAD_TOKEN:"PAD",
        TO_JV_TOKEN:TO_JV_TOKEN_STR,
        TO_SU_TOKEN:TO_SU_TOKEN_STR
    }
    self.num_words = 5
  def get_name(self)->str:
      return self.name

  def add_sentence(self, sentence:str)-> None:
    for word in sentence.split(" "):
      self.add_word(word)

  def add_word(self, word:str)->None:
    if word not in self.word_to_index:
      self.word_to_index[word]= self.num_words
      self.index_to_word[self.num_words]=word
      self.num_words += 1
    self.word_to_count[word] += 1