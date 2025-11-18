from src.config import MAX_LENGTH
from src.data.data_reader import DataReader
from src.data.language_data import LanguageData


class Preprocessor:
  def __init__(self,first_language=LanguageData,second_language=LanguageData,type="train"):
    self.input_lang = first_language
    self.output_lang = second_language
    self.type_data = type

  def process(self)-> tuple[LanguageData,LanguageData,list[list[str]]]:

      # Initialize variables to safe defaults

    all_pairs = []

    reader = DataReader(self.input_lang, self.output_lang, type_data=self.type_data)

    if self.type_data == "train":
       input_lang, output_lang, all_pairs = reader.read_train()
    elif self.type_data == "valid":
        input_lang, output_lang, all_pairs = reader.read_valid()
    elif self.type_data == "test":
        input_lang, output_lang, all_pairs = reader.read_test()
    else:
        # Handle unexpected type_data gracefully
        print(f"Warning: Unknown data type '{self.type_data}'. No pairs loaded.")

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