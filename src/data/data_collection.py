from src.data.language_data import LanguageData
from src.data.pre_processor import Preprocessor


class DataCollection:
    def __init__(self):
        self.combined_output_language = None
        self.combined_input_language = None
        self.combined_valid_pairs = None
        self.combined_train_pairs = None
        _, _, self.pairs_train_data_id_jv = Preprocessor(first_language="id", second_language="jv", type="train").process()
        _, _, self.pairs_valid_data_id_jv = Preprocessor(first_language="id", second_language="jv", type="valid").process()

        _, _, self.pairs_train_data_id_su = Preprocessor(first_language="id", second_language="su", type="train").process()
        _, _, self.pairs_valid_data_id_su = Preprocessor(first_language="id", second_language="su", type="valid").process()
        self.populate()

    def populate(self):
        self.combined_input_language = LanguageData("id")
        self.combined_output_language = LanguageData("multi")  # "multi" to indicate it holds Javanese and Sundanese

        # Populate combined vocabularies
        for pair in self.pairs_train_data_id_jv:
            self.combined_input_language.add_sentence(pair[0])
            self.combined_output_language.add_sentence(pair[1])

        for pair in self.pairs_valid_data_id_jv:
            self.combined_input_language.add_sentence(pair[0])
            self.combined_output_language.add_sentence(pair[1])

        for pair in self.pairs_train_data_id_su:
            self.combined_input_language.add_sentence(pair[0])
            self.combined_output_language.add_sentence(pair[1])

        for pair in self.pairs_valid_data_id_su:
            self.combined_input_language.add_sentence(pair[0])
            self.combined_output_language.add_sentence(pair[1])

        print(
            f"Combined Input Language Name: {self.combined_input_language.name}, Number of words: {self.combined_input_language.num_words}")
        print(
            f"Combined Output Language Name: {self.combined_output_language.name}, Number of words: {self.combined_output_language.num_words}")

        # Concatenate pairs
        self.combined_train_pairs = self.pairs_train_data_id_jv + self.pairs_train_data_id_su
        self.combined_valid_pairs = self.pairs_valid_data_id_jv + self.pairs_valid_data_id_su

        print(f"Combined training pairs count: {len(self.combined_train_pairs)}")
        print(f"Combined validation pairs count: {len(self.combined_valid_pairs)}")
    def get_combined_input_language(self)-> LanguageData:
        return self.combined_input_language

    def get_combined_output_language(self)-> LanguageData:
        return self.combined_output_language
    
    def get_train_pairs(self)->list:
        return self.combined_train_pairs
    
    def get_validation_pairs(self)->list:
        return self.combined_valid_pairs