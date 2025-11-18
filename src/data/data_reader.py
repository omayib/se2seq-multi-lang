from src.config import TO_JV_TOKEN_STR, TO_SU_TOKEN_STR
from src.data.fetch_raw_data import load_nusax_datasets
from src.data.language_data import LanguageData
from src.data.utils import normalize_string


class DataReader:
  def __init__(self, first_language:str="id", second_language:str="jv", type_data="train")->None:

    raw_dataset = load_nusax_datasets()
    self.first_language = first_language
    self.second_language = second_language

    # Select the correct dataframe based on type_data and second_language
    if type_data == "train":
        if second_language == "jv":
            self.current_train_df = raw_dataset["jv_train"]
        elif second_language == "su":
            self.current_train_df = raw_dataset["su_train"]
        else:
            raise ValueError("Unsupported second_language for training data.")

        self.pairs_train = self.current_train_df.to_numpy().tolist()

        self.pairs_train_normalized = [
            [normalize_string(string) for string in pair] for pair in self.pairs_train
        ]

        for i, pair in enumerate(self.pairs_train_normalized):
            if self.second_language == "jv":
                self.pairs_train_normalized[i][0] = f"{TO_JV_TOKEN_STR} {pair[0]}"
            elif self.second_language == "su":
                self.pairs_train_normalized[i][0] = f"{TO_SU_TOKEN_STR} {pair[0]}"
            print(f"DataReader {self.second_language} pari : {self.pairs_train_normalized[i]}")

    elif type_data == "valid":
        if second_language == "jv":
            self.current_valid_df = raw_dataset["jv_valid"]
        elif second_language == "su":
            self.current_valid_df = raw_dataset["su_valid"]
        else:
            raise ValueError("Unsupported second_language for validation data.")

        self.pairs_valid = self.current_valid_df.to_numpy().tolist()
        self.pairs_valid_normalized = [
            [normalize_string(string) for string in pair] for pair in self.pairs_valid
        ]

        for i, pair in enumerate(self.pairs_valid_normalized):
            if self.second_language == "jv":
                self.pairs_valid_normalized[i][0] = f"{TO_JV_TOKEN_STR} {pair[0]}"
            elif self.second_language == "su":
                self.pairs_valid_normalized[i][0] = f"{TO_SU_TOKEN_STR} {pair[0]}"

    elif type_data == "test":
        if second_language == "jv":
            self.current_test_df = raw_dataset["jv_test"]
        elif second_language == "su":
            self.current_test_df = raw_dataset["su_test"]
        else:
            raise ValueError("Unsupported second_language for test data.")
        self.pairs_test = self.current_test_df.to_numpy().tolist()
        self.pairs_test_normalized = [
            [normalize_string(string) for string in pair] for pair in self.pairs_test
        ]

        # Prepend language tokens to the input sentences

        for i, pair in enumerate(self.pairs_test_normalized):
            if self.second_language == "jv":
                self.pairs_test_normalized[i][0] = f"{TO_JV_TOKEN_STR} {pair[0]}"
            elif self.second_language == "su":
                self.pairs_test_normalized[i][0] = f"{TO_SU_TOKEN_STR} {pair[0]}"

    else:
        raise ValueError("Unsupported type_data.")


  def read_train(self)->tuple[LanguageData, LanguageData, list[list[str]]]:
    input_language = LanguageData(self.first_language)
    output_language = LanguageData(self.second_language)
    return input_language, output_language, self.pairs_train_normalized

  def read_valid(self)->tuple[LanguageData, LanguageData, list[list[str]]]:
    input_language = LanguageData(self.first_language)
    output_language = LanguageData(self.second_language)
    return input_language, output_language, self.pairs_valid_normalized

  def read_test(self)->tuple[LanguageData, LanguageData, list[list[str]]]:
    input_language = LanguageData(self.first_language)
    output_language = LanguageData(self.second_language)
    return input_language, output_language, self.pairs_test_normalized
