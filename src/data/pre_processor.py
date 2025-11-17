from src.config import MAX_LENGTH
from src.data.data_reader import DataReader
from src.data.language_data import LanguageData


class Preprocessor:
  def __init__(self,first_language="id",second_language="jv",type="train"):
    self.first_language = first_language
    self.second_language = second_language
    self.type_data = type
  def process(self)-> tuple[LanguageData,LanguageData,list[list[str]]]:
    input_lang, output_lang, all_pairs = DataReader(self.first_language, self.second_language,type_data=self.type_data).read()

    for pair in all_pairs:
      input_lang.add_sentence(pair[0])
      output_lang.add_sentence(pair[1])

    filtered_pairs = []
    initial_pairs_count = len(all_pairs)
    for pair in all_pairs:
        if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH:
            filtered_pairs.append(pair)

    print(f"Filtered {initial_pairs_count - len(filtered_pairs)} pairs out of {initial_pairs_count} due to exceeding MAX_LENGTH ({MAX_LENGTH}).")
    return input_lang,output_lang, filtered_pairs